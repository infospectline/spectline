
import { state } from "./state.js";
import {
    timeCoord,
    priceCoord,
    getChartContainer,
    getMainPlotRect,
    isInsideMainPlot,
    geometryXToTime,
    geometryYToPrice
} from "./chart.js";


// ============================================================================
// MODULE STATE
// ============================================================================

const elements = {
    marketSelect: null,
    timeframeSelect: null,
    learningMarket: null,
    learningTimeframe: null,
    learningTradeCount: null,
    learningView: null,
    resultsView: null,
    learningMainView: null,
    learningListView: null,
    learningDetailView: null,
    learningTradeList: null,
    learningSaveButton: null,
    learningDeleteButton: null,
    learningAddButton: null,
    learningAnchorTime: null,
    learningAnchorPrice: null,
    learningAnchorPicker: null,
    learningDetailTradeNumber: null,
    learningSetupTimeframe: null,
    learningEntryTimeframe: null,
    learningRRModeButtons: [],
    deleteAnchorButton: null,
    deleteTradeSetupButton: null,
    drawTradeSetupButton: null,
    addZoneButton: null,
    zoneTypeMenu: null,
    addSupportButton: null,
    addResistanceButton: null,
    expandAnchorButton: null,
    expandTradeSetupButton: null,
    sidebar: null,
    sidebarToggle: null
};

let callbacks = {
    onSendAction: null,
    onEnsureGeometry: null,
    onRenderGeometry: null,
    onUpdateGeometrySummary: null,
    onSetGeometryTool: null,
    onClearTradeSetup: null,
    onCancelInteraction: null
};

let listenerController = null;
let initialized = false;


// ============================================================================
// INTERNAL HELPERS
// ============================================================================

function getElement(id) {
    return document.getElementById(id);
}

function isFiniteNumber(value) {
    return (
        value !== null &&
        value !== undefined &&
        value !== "" &&
        Number.isFinite(Number(value))
    );
}

function isMobileTouchDevice() {
    return window.matchMedia(
        "(max-width: 900px) and (pointer: coarse)"
    ).matches;
}

function getOverlay() {
    return document.getElementById("overlay");
}

function getSelectedLearningTrade() {
    if (state.learning.selectedTradeId === null) {
        return null;
    }

    return state.learning.trades.find(
        trade => trade.id === state.learning.selectedTradeId
    ) || null;
}

function getSidebarIcon() {
    return elements.sidebarToggle?.querySelector("img") || null;
}

function setSidebarIcon(source) {
    const image = getSidebarIcon();

    if (image) {
        image.src = source;
    }
}

function callCallback(name, ...args) {
    const callback = callbacks[name];

    if (typeof callback !== "function") {
        return undefined;
    }

    return callback(...args);
}

function formatPrice(price) {
    if (!isFiniteNumber(price)) {
        return "-";
    }

    return Number(price).toFixed(5);
}

function formatAnchorTime(timestamp) {
    if (!isFiniteNumber(timestamp)) {
        return "";
    }

    const date = new Date(Number(timestamp) * 1000);

    if (Number.isNaN(date.getTime())) {
        return "";
    }

    return date.toISOString()
        .replace("T", " ")
        .replace(".000Z", " UTC");
}

function normalizeTradeId(value) {
    return value;
}

function ensureTradeGeometry(trade) {
    if (!trade) {
        return null;
    }

    callCallback("onEnsureGeometry", trade);
    return ensureLearningTrade(trade);
}

function createEmptyTradeGeometry() {
    return {
        start_time: null,
        end_time: null,
        breakout: {
            time: null,
            price: null
        },
        entry: {
            time: null,
            price: null
        },
        sl: {
            time: null,
            price: null
        },
        tp: {
            time: null,
            price: null
        }
    };
}

function clearPickerButtonStates() {
    document
        .querySelectorAll(".learning-price-picker.active")
        .forEach(button => button.classList.remove("active"));
}

function updateAnchorButtonState() {
    const trade = getSelectedLearningTrade();

    const hasAnchor =
        trade?.anchor?.time != null &&
        trade?.anchor?.price != null;

    elements.learningAnchorPicker?.classList.toggle(
        "has-anchor",
        Boolean(hasAnchor)
    );
}

function setLearningViewVisibility({
    learning = null,
    results = null,
    main = null,
    list = null,
    detail = null
} = {}) {
    if (learning !== null && elements.learningView) {
        elements.learningView.style.display = learning ? "block" : "none";
    }

    if (results !== null && elements.resultsView) {
        elements.resultsView.style.display = results ? "block" : "none";
    }

    if (main !== null && elements.learningMainView) {
        elements.learningMainView.style.display = main ? "block" : "none";
    }

    if (list !== null && elements.learningListView) {
        elements.learningListView.style.display = list ? "block" : "none";
    }

    if (detail !== null && elements.learningDetailView) {
        elements.learningDetailView.style.display = detail ? "block" : "none";
    }
}

function notifyGeometryRefresh() {
    callCallback("onRenderGeometry");
    callCallback("onUpdateGeometrySummary");
}

function sendLearningAction(payload) {
    if (typeof callbacks.onSendAction !== "function") {
        console.error("Learning action handler is not configured.");
        return false;
    }

    try {
        return callbacks.onSendAction(payload) !== false;
    } catch (error) {
        console.error("Failed to send learning action:", error);
        return false;
    }
}

function bindClick(element, handler, signal) {
    if (!element) {
        return;
    }

    element.addEventListener("click", handler, { signal });
}

function preventEvent(event) {
    event.preventDefault();
    event.stopPropagation();
}


// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

export function initializeLearning({
    onSendAction = null,
    onEnsureGeometry = null,
    onRenderGeometry = null,
    onUpdateGeometrySummary = null,
    onSetGeometryTool = null,
    onClearTradeSetup = null,
    onCancelInteraction = null
} = {}) {
    if (listenerController) {
        listenerController.abort();
        listenerController = null;
    }

    callbacks = {
        onSendAction,
        onEnsureGeometry,
        onRenderGeometry,
        onUpdateGeometrySummary,
        onSetGeometryTool,
        onClearTradeSetup,
        onCancelInteraction
    };

    elements.marketSelect = getElement("marketSelect");
    elements.timeframeSelect = getElement("timeframeSelect");
    elements.learningMarket = getElement("learningMarket");
    elements.learningTimeframe = getElement("learningTimeframe");
    elements.learningTradeCount = getElement("learningTradeCount");
    elements.learningView = getElement("learningView");
    elements.resultsView = getElement("resultsView");
    elements.learningMainView = getElement("learningMainView");
    elements.learningListView = getElement("learningListView");
    elements.learningDetailView = getElement("learningDetailView");
    elements.learningTradeList = getElement("learningTradeList");
    elements.learningSaveButton = getElement("learningSaveButton");
    elements.learningDeleteButton = getElement("learningDeleteButton");
    elements.learningAddButton = getElement("learningAddButton");
    elements.learningAnchorTime = getElement("learningAnchorTime");
    elements.learningAnchorPrice = getElement("learningAnchorPrice");
    elements.learningAnchorPicker = getElement("learningAnchorPicker");
    elements.learningDetailTradeNumber = getElement("learningDetailTradeNumber");
    elements.learningSetupTimeframe = getElement("learningSetupTimeframe");
    elements.learningEntryTimeframe = getElement("learningEntryTimeframe");
    elements.learningRRModeButtons = Array.from(
        document.querySelectorAll(".learning-rr-mode-button")
    );
    elements.deleteAnchorButton = getElement("deleteAnchorButton");
    elements.deleteTradeSetupButton = getElement("deleteTradeSetupButton");
    elements.drawTradeSetupButton = getElement("drawTradeSetupButton");
    elements.addZoneButton = getElement("addZoneButton");
    elements.zoneTypeMenu = getElement("zoneTypeMenu");
    elements.addSupportButton = getElement("addSupportButton");
    elements.addResistanceButton = getElement("addResistanceButton");
    elements.expandAnchorButton = getElement("expandAnchorButton");
    elements.expandTradeSetupButton = getElement("expandTradeSetupButton");
    elements.sidebar = getElement("sidebar");
    elements.sidebarToggle = getElement("sidebarToggle");

    listenerController = new AbortController();

    bindLearningEvents(listenerController.signal);
    initialized = true;

    updateLearningHeader();
    renderLearningTradeList();
    updateAnchorButtonState();

    return true;
}


// ============================================================================
// LEARNING DATA
// ============================================================================

export function setLearningTrades(trades) {
    if (!Array.isArray(trades)) {
        return false;
    }

    state.learning.trades = trades;

    state.learning.nextTradeId =
        state.learning.trades.reduce(
            (maxId, trade) => Math.max(maxId, Number(trade?.id) || 0),
            0
        ) + 1;

    renderLearningTradeList();

    const selectedTrade = getSelectedLearningTrade();

    if (selectedTrade) {
        ensureTradeGeometry(selectedTrade);
        notifyGeometryRefresh();
        renderLearningAnchor();
    }

    return true;
}

export function getLearningTrades() {
    return state.learning.trades;
}

export function getSelectedLearningTradeData() {
    return getSelectedLearningTrade();
}

export function ensureLearningTrade(trade) {
    if (!trade || typeof trade !== "object") {
        return null;
    }

    if (!Array.isArray(trade.zones)) {
        trade.zones = [];
    }

    if (!trade.trade_geometry) {
        trade.trade_geometry = createEmptyTradeGeometry();
    }

    return trade;
}


// ============================================================================
// LEARNING TRADE LIST
// ============================================================================

export function renderLearningTradeList() {
    const list = elements.learningTradeList || getElement("learningTradeList");
    const count = elements.learningTradeCount || getElement("learningTradeCount");

    if (!list) {
        return false;
    }

    list.replaceChildren();

    const trades = [...state.learning.trades];

    if (count) {
        count.textContent = String(trades.length);
    }

    if (!trades.length) {
        const empty = document.createElement("div");
        empty.className = "learning-trade-empty";
        empty.textContent = "NO LEARNING TRADES";
        list.appendChild(empty);
        return true;
    }

    const fragment = document.createDocumentFragment();

    trades.forEach((trade, index) => {
        const button = document.createElement("button");

        button.type = "button";
        button.className = "learning-trade-item";
        button.textContent = `${index + 1}. ${trade.direction || "LONG"}`;
        button.dataset.tradeId = String(trade.id);

        button.addEventListener("click", () => {
            openLearningTradeDetail(trade.id);
        });

        fragment.appendChild(button);
    });

    list.appendChild(fragment);
    return true;
}


// ============================================================================
// LEARNING TRADE DETAIL
// ============================================================================

export function openLearningTradeDetail(tradeId) {
    const trade = state.learning.trades.find(
        item => item.id === normalizeTradeId(tradeId)
    );

    if (!trade) {
        return false;
    }

    callCallback("onCancelInteraction");
    cancelPricePicker();
    cancelAnchorPicker();
    callCallback("onSetGeometryTool", null);

    elements.drawTradeSetupButton?.classList.remove("selected");

    state.learning.selectedTradeId = trade.id;

    const tradeIndex = state.learning.trades.findIndex(
        item => item.id === trade.id
    );

    if (elements.learningDetailTradeNumber) {
        elements.learningDetailTradeNumber.textContent = String(tradeIndex + 1);
    }

    if (elements.learningSetupTimeframe) {
        elements.learningSetupTimeframe.value = trade.setup_timeframe || "H1";
    }

    if (elements.learningEntryTimeframe) {
        elements.learningEntryTimeframe.value = trade.entry_timeframe || "M15";
    }

    if (elements.learningAnchorTime) {
        elements.learningAnchorTime.value = trade.anchor?.time
            ? formatAnchorTime(trade.anchor.time)
            : "";
    }

    if (elements.learningAnchorPrice) {
        elements.learningAnchorPrice.value = trade.anchor?.price ?? "";
    }

    const rrMode = trade.rr_mode || "OFF";

    elements.learningRRModeButtons.forEach(button => {
        button.classList.toggle(
            "active",
            button.dataset.rrMode === rrMode
        );
    });

    ensureTradeGeometry(trade);

    setLearningViewVisibility({
        learning: true,
        results: false,
        main: false,
        list: false,
        detail: true
    });

    elements.sidebar?.classList.remove("collapsed");
    state.learning.pendingAnchorCenter = Boolean(trade.anchor?.time);

    updateAnchorButtonState();

    requestAnimationFrame(() => {
        renderLearningAnchor();
        notifyGeometryRefresh();
        centerLearningChartOnAnchor();
    });

    setSidebarIcon("images/left.png");

    return true;
}

export function closeLearningTradeDetail() {
    callCallback("onCancelInteraction");
    cancelPricePicker();
    cancelAnchorPicker();
    callCallback("onSetGeometryTool", null);

    if (state.learning.selectedTradeIsNew) {
        state.learning.trades = state.learning.trades.filter(
            trade => trade.id !== state.learning.selectedTradeId
        );
    }

    state.learning.selectedTradeId = null;
    state.learning.selectedTradeIsNew = false;

    setLearningViewVisibility({
        detail: false,
        list: true
    });

    renderLearningTradeList();
    notifyGeometryRefresh();

    return true;
}

export function createLearningTrade() {
    const trade = {
        id: state.learning.nextTradeId++,
        direction: null,
        setup_timeframe: elements.timeframeSelect?.value || "H1",
        entry_timeframe: "M15",
        anchor: {
            time: null,
            price: null
        },
        rr_mode: "OFF",
        zones: [],
        trade_geometry: createEmptyTradeGeometry(),
        breakout: null,
        entry_candle: "",
        entry: null,
        sl: null,
        tp: null
    };

    state.learning.trades.push(trade);
    state.learning.selectedTradeIsNew = true;

    openLearningTradeDetail(trade.id);

    return trade;
}

export function saveLearningTrade() {
    const trade = getSelectedLearningTrade();

    if (!trade) {
        return false;
    }

    if (elements.learningSetupTimeframe) {
        trade.setup_timeframe = elements.learningSetupTimeframe.value;
    }

    if (elements.learningEntryTimeframe) {
        trade.entry_timeframe = elements.learningEntryTimeframe.value;
    }

    ensureTradeGeometry(trade);

    trade.anchor = {
        time: trade.anchor?.time ?? null,
        price: trade.anchor?.price ?? null
    };

    if (trade.trade_geometry) {
        trade.breakout = trade.trade_geometry.breakout?.price ?? null;
        trade.entry = trade.trade_geometry.entry?.price ?? null;
        trade.sl = trade.trade_geometry.sl?.price ?? null;
        trade.tp = trade.trade_geometry.tp?.price ?? null;
    }

    const sent = sendLearningAction({
        action: "learning_save",
        setup: trade
    });

    if (!sent) {
        return false;
    }

    state.learning.selectedTradeIsNew = false;
    closeLearningTradeDetail();

    return true;
}

export function deleteLearningTrade() {
    const tradeId = state.learning.selectedTradeId;

    if (tradeId === null) {
        return false;
    }

    const sent = sendLearningAction({
        action: "learning_delete",
        id: tradeId
    });

    if (!sent) {
        return false;
    }

    callCallback("onCancelInteraction");
    cancelPricePicker();
    cancelAnchorPicker();
    callCallback("onSetGeometryTool", null);

    state.learning.selectedTradeId = null;
    state.learning.selectedTradeIsNew = false;

    setLearningViewVisibility({
        detail: false,
        list: true
    });

    renderLearningTradeList();
    notifyGeometryRefresh();

    return true;
}


// ============================================================================
// LEARNING NAVIGATION
// ============================================================================

export function updateLearningHeader() {
    if (elements.learningMarket) {
        elements.learningMarket.textContent = elements.marketSelect?.value || "";
    }

    if (elements.learningTimeframe) {
        elements.learningTimeframe.textContent = elements.timeframeSelect?.value || "";
    }
}

export function openLearningMode() {
    state.learning.mode = true;

    callCallback("onCancelInteraction");
    cancelPricePicker();
    cancelAnchorPicker();
    callCallback("onSetGeometryTool", null);

    elements.sidebar?.classList.remove("collapsed");

    setLearningViewVisibility({
        learning: true,
        results: false,
        main: true,
        list: false,
        detail: false
    });

    setSidebarIcon("images/left.png");
    updateLearningHeader();
    renderLearningTradeList();

    sendLearningAction({ action: "learning_load" });

    return true;
}

export function openLearningListView() {
    callCallback("onCancelInteraction");
    cancelPricePicker();
    cancelAnchorPicker();
    callCallback("onSetGeometryTool", null);

    setLearningViewVisibility({
        learning: true,
        main: false,
        list: true,
        detail: false
    });

    renderLearningTradeList();

    return true;
}

export function closeLearningMode() {
    state.learning.mode = false;

    callCallback("onCancelInteraction");
    cancelPricePicker();
    cancelAnchorPicker();
    callCallback("onSetGeometryTool", null);

    setLearningViewVisibility({
        learning: false,
        results: true,
        detail: false
    });

    elements.sidebar?.classList.remove("collapsed");
    setSidebarIcon("images/right.png");

    return true;
}


// ============================================================================
// ANCHOR AND PRICE PICKERS
// ============================================================================

export function cancelPricePicker() {
    state.interaction.activePriceTarget = null;

    clearPickerButtonStates();

    getChartContainer()?.classList.remove("price-picker-active");

    return true;
}

export function activatePricePicker(targetId, button = null) {
    if (!targetId || !getChartContainer()) {
        return false;
    }

    callCallback("onCancelInteraction");
    cancelAnchorPicker();
    cancelPricePicker();

    state.interaction.activePriceTarget = targetId;

    const activeButton = button || Array.from(
        document.querySelectorAll(".learning-price-picker")
    ).find(item => item.dataset.priceTarget === String(targetId));

    activeButton?.classList.add("active");
    getChartContainer().classList.add("price-picker-active");

    return true;
}

export function pickPriceFromChart(event) {
    const targetId = state.interaction.activePriceTarget;

    if (!targetId || !event) {
        return false;
    }

    const container = getChartContainer();
    const rect = container?.getBoundingClientRect();

    if (!container || !rect) {
        cancelPricePicker();
        return false;
    }

    if (
        isMobileTouchDevice() &&
        !isInsideMainPlot(event.clientX, event.clientY)
    ) {
        return false;
    }

    const y = event.clientY - rect.top;
    const price = geometryYToPrice(y);

    if (!isFiniteNumber(price)) {
        return false;
    }

    const input = getElement(targetId);

    if (!input) {
        cancelPricePicker();
        return false;
    }

    input.value = Number(price).toFixed(5);
    input.dispatchEvent(new Event("input", { bubbles: true }));

    cancelPricePicker();

    return true;
}

export function syncAnchorButtonState() {
    updateAnchorButtonState();
}

export function cancelAnchorPicker() {
    state.interaction.anchorPickerActive = false;

    elements.learningAnchorPicker?.classList.remove("active");
    updateAnchorButtonState();

    getChartContainer()?.classList.remove("price-picker-active");

    return true;
}

export function activateAnchorPicker() {
    if (!getChartContainer()) {
        return false;
    }

    callCallback("onCancelInteraction");
    cancelPricePicker();
    cancelAnchorPicker();

    state.interaction.anchorPickerActive = true;

    elements.learningAnchorPicker?.classList.add("active");
    elements.learningAnchorPicker?.classList.remove("has-anchor");
    getChartContainer().classList.add("price-picker-active");

    return true;
}

export function pickAnchorFromChart(event) {
    if (
        !state.interaction.anchorPickerActive ||
        !event ||
        state.learning.selectedTradeId === null
    ) {
        return false;
    }

    const container = getChartContainer();
    const rect = container?.getBoundingClientRect();

    if (!container || !rect) {
        cancelAnchorPicker();
        return false;
    }

    if (
        isMobileTouchDevice() &&
        !isInsideMainPlot(event.clientX, event.clientY)
    ) {
        return false;
    }

    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;

    if (
        localX < 0 ||
        localX > rect.width ||
        localY < 0 ||
        localY > rect.height
    ) {
        return false;
    }

    const timestamp = geometryXToTime(localX);
    const price = geometryYToPrice(localY);

    if (!isFiniteNumber(timestamp) || !isFiniteNumber(price)) {
        return false;
    }

    const trade = getSelectedLearningTrade();

    if (!trade) {
        return false;
    }

    trade.anchor = {
        time: Math.floor(Number(timestamp)),
        price: Number(price)
    };

    if (elements.learningAnchorTime) {
        elements.learningAnchorTime.value = formatAnchorTime(trade.anchor.time);
    }

    if (elements.learningAnchorPrice) {
        elements.learningAnchorPrice.value = Number(price).toFixed(5);
    }

    state.learning.pendingAnchorCenter = false;

    cancelAnchorPicker();
    renderLearningAnchor();

    return true;
}

export function renderLearningAnchor() {
    const overlay = getOverlay();

    if (!overlay) {
        return false;
    }

    overlay.querySelectorAll(
        ".learning-anchor-point, .learning-anchor-label"
    ).forEach(element => element.remove());

    const trade = getSelectedLearningTrade();

    if (
        !trade ||
        !trade.anchor ||
        trade.anchor.time == null ||
        trade.anchor.price == null
    ) {
        updateAnchorButtonState();
        return false;
    }

    const x = timeCoord(trade.anchor.time);
    const y = priceCoord(trade.anchor.price);

    if (x === null || y === null) {
        return false;
    }

    let markerX = x;
    let markerY = y;

    if (isMobileTouchDevice()) {
        const container = getChartContainer();
        const containerRect = container?.getBoundingClientRect();
        const plotRect = getMainPlotRect();

        if (!containerRect || !plotRect) {
            return false;
        }

        markerX += plotRect.left - containerRect.left;
        markerY += plotRect.top - containerRect.top;
    }

    const point = document.createElement("div");
    point.className = "learning-anchor-point";
    point.style.left = `${markerX}px`;
    point.style.top = `${markerY}px`;
    overlay.appendChild(point);

    const label = document.createElement("div");
    label.className = "learning-anchor-label";
    label.textContent = `ANCHOR ${formatPrice(trade.anchor.price)}`;
    label.style.left = `${markerX + 10}px`;
    label.style.top = `${markerY - 22}px`;
    overlay.appendChild(label);

    updateAnchorButtonState();

    return true;
}

export function centerLearningChartOnAnchor() {
    if (!state.learning.pendingAnchorCenter) {
        return false;
    }

    const trade = getSelectedLearningTrade();
    const chart = state.chart.instance;

    if (
        !trade ||
        !trade.anchor ||
        trade.anchor.time == null ||
        !chart
    ) {
        state.learning.pendingAnchorCenter = false;
        return false;
    }

    const anchorTime = Number(trade.anchor.time);

    if (!Number.isFinite(anchorTime)) {
        state.learning.pendingAnchorCenter = false;
        return false;
    }

    const timeScale = chart.timeScale();
    const visibleRange = timeScale.getVisibleRange();

    let span = 60 * 60 * 24;

    if (
        visibleRange &&
        isFiniteNumber(visibleRange.from) &&
        isFiniteNumber(visibleRange.to)
    ) {
        span = Number(visibleRange.to) - Number(visibleRange.from);

        if (!Number.isFinite(span) || span <= 0) {
            span = 60 * 60 * 24;
        }
    }

    const halfSpan = span / 2;

    try {
        timeScale.setVisibleRange({
            from: anchorTime - halfSpan,
            to: anchorTime + halfSpan
        });
    } catch (error) {
        console.warn("Unable to center chart on learning anchor:", error);
        state.learning.pendingAnchorCenter = false;
        return false;
    }

    state.learning.pendingAnchorCenter = false;
    requestAnimationFrame(renderLearningAnchor);

    return true;
}


// ============================================================================
// LEARNING ZONE LIST
// ============================================================================

function normalizeZoneGeometry(rect) {
    if (!rect) {
        return null;
    }

    const startTime = Number(rect.start_time);
    const endTime = Number(rect.end_time);
    const low = Number(rect.low);
    const high = Number(rect.high);

    if (
        !Number.isFinite(startTime) ||
        !Number.isFinite(endTime) ||
        !Number.isFinite(low) ||
        !Number.isFinite(high)
    ) {
        return null;
    }

    return {
        start_time: Math.min(startTime, endTime),
        end_time: Math.max(startTime, endTime),
        low: Math.min(low, high),
        high: Math.max(low, high)
    };
}

function getZoneColor(zoneType) {
    if (zoneType === "SUPPORT") {
        return "#00d26a";
    }

    if (zoneType === "RESISTANCE") {
        return "#ff3b30";
    }

    return "#aaaaaa";
}

export function renumberLearningZones(trade = getSelectedLearningTrade()) {
    if (!trade) {
        return;
    }

    let supportNumber = 0;
    let resistanceNumber = 0;

    for (const zone of trade.zones || []) {
        zone.color = getZoneColor(zone.type);

        if (zone.type === "SUPPORT") {
            supportNumber++;
            zone.number = supportNumber;
        } else if (zone.type === "RESISTANCE") {
            resistanceNumber++;
            zone.number = resistanceNumber;
        }
    }
}

export function renderLearningZoneList() {
    const list = getElement("learningZoneList");
    const trade = getSelectedLearningTrade();

    if (!list) {
        return false;
    }

    list.replaceChildren();

    if (!trade) {
        return true;
    }

    ensureLearningTrade(trade);
    renumberLearningZones(trade);

    const fragment = document.createDocumentFragment();

    for (const zone of trade.zones) {
        const item = document.createElement("div");
        item.className = "learning-zone-item";
        item.dataset.zoneId = String(zone.id);

        const header = document.createElement("div");
        header.className = "learning-zone-header";

        const mainButton = document.createElement("button");
        mainButton.type = "button";
        mainButton.className = "learning-zone-main";
        mainButton.textContent = `${zone.type} ${zone.number}`;
        mainButton.style.borderLeft = `3px solid ${zone.color}`;

        const expandButton = document.createElement("button");
        expandButton.type = "button";
        expandButton.className = "learning-zone-expand";
        expandButton.textContent = "▼";

        const deleteButton = document.createElement("button");
        deleteButton.type = "button";
        deleteButton.className = "learning-zone-delete";
        deleteButton.textContent = "×";
        deleteButton.title = `Delete ${zone.type} ${zone.number}`;

        const body = document.createElement("div");
        body.className = "learning-zone-body";

        const geometry = normalizeZoneGeometry(zone.geometry);

        body.textContent = geometry
            ? `${formatAnchorTime(geometry.start_time)} → ${formatAnchorTime(geometry.end_time)} | ${formatPrice(geometry.low)} - ${formatPrice(geometry.high)}`
            : "Not set";

        mainButton.addEventListener("click", () => {
            callCallback("onSetGeometryTool", "zone", zone.type);
        });

        expandButton.addEventListener("click", event => {
            preventEvent(event);

            const isOpen = item.classList.toggle("open");
            expandButton.textContent = isOpen ? "▲" : "▼";
        });

        deleteButton.addEventListener("click", event => {
            preventEvent(event);
            callCallback("onCancelInteraction");

            trade.zones = trade.zones.filter(
                current => current.id !== zone.id
            );

            renumberLearningZones(trade);
            callCallback("onSetGeometryTool", null);
            renderLearningZoneList();
            notifyGeometryRefresh();
        });

        header.append(mainButton, expandButton, deleteButton);
        item.append(header, body);
        fragment.appendChild(item);
    }

    list.appendChild(fragment);
    return true;
}


// ============================================================================
// GEOMETRY SUMMARY
// ============================================================================

export function updateLearningGeometrySummary() {
    const trade = getSelectedLearningTrade();

    if (!trade) {
        return false;
    }

    ensureLearningTrade(trade);
    renumberLearningZones(trade);
    renderLearningZoneList();

    const setup = trade.trade_geometry;

    const summary = getElement("tradeSetupSummary");
    const breakoutValue = getElement("learningBreakoutValue");
    const entryValue = getElement("learningEntryValue");
    const slValue = getElement("learningSLValue");
    const tpValue = getElement("learningTPValue");

    if (summary) {
        summary.textContent =
            setup?.start_time != null && setup?.end_time != null
                ? `${formatAnchorTime(setup.start_time)} → ${formatAnchorTime(setup.end_time)}`
                : "Not set";
    }

    if (breakoutValue) {
        breakoutValue.textContent = setup?.breakout?.price != null
            ? `${formatPrice(setup.breakout.price)} | ${formatAnchorTime(setup.breakout.time)}`
            : "-";
    }

    if (entryValue) {
        entryValue.textContent = setup?.entry?.price != null
            ? `${formatPrice(setup.entry.price)} | ${formatAnchorTime(setup.entry.time)}`
            : "-";
    }

    if (slValue) {
        slValue.textContent = setup?.sl?.price != null
            ? `${formatPrice(setup.sl.price)} | ${formatAnchorTime(setup.sl.time)}`
            : "-";
    }

    if (tpValue) {
        tpValue.textContent = setup?.tp?.price != null
            ? `${formatPrice(setup.tp.price)} | ${formatAnchorTime(setup.tp.time)}`
            : "-";
    }

    return true;
}


// ============================================================================
// EVENT LISTENERS
// ============================================================================

function setupAccordion(button, signal) {
    bindClick(button, event => {
        preventEvent(event);

        const item = button.closest(".learning-accordion-item");

        if (!item) {
            return;
        }

        const isOpen = item.classList.toggle("open");
        button.textContent = isOpen ? "▲" : "▼";
    }, signal);
}

function bindLearningEvents(signal) {
    bindClick(elements.learningAddButton, () => {
        createLearningTrade();
    }, signal);

    bindClick(elements.learningSaveButton, () => {
        saveLearningTrade();
    }, signal);

    bindClick(elements.learningDeleteButton, () => {
        deleteLearningTrade();
    }, signal);

    bindClick(getElement("learningButton"), () => {
        openLearningListView();
    }, signal);

    bindClick(getElement("learningTabButton"), () => {
        openLearningMode();
    }, signal);

    bindClick(getElement("resultsTabButton"), () => {
        closeLearningMode();
    }, signal);

    setupAccordion(elements.expandAnchorButton, signal);
    setupAccordion(elements.expandTradeSetupButton, signal);

    bindClick(elements.drawTradeSetupButton, () => {
        callCallback("onSetGeometryTool", "trade");
        elements.drawTradeSetupButton.classList.add("selected");
    }, signal);

    bindClick(elements.addZoneButton, event => {
        preventEvent(event);

        if (!elements.zoneTypeMenu) {
            return;
        }

        const isVisible = elements.zoneTypeMenu.style.display !== "none";

        elements.zoneTypeMenu.style.display = isVisible ? "none" : "flex";
    }, signal);

    bindClick(elements.addSupportButton, event => {
        preventEvent(event);
        if (elements.zoneTypeMenu) {
            elements.zoneTypeMenu.style.display = "none";
        }
        callCallback("onSetGeometryTool", "zone", "SUPPORT");
    }, signal);

    bindClick(elements.addResistanceButton, event => {
        preventEvent(event);
        if (elements.zoneTypeMenu) {
            elements.zoneTypeMenu.style.display = "none";
        }
        callCallback("onSetGeometryTool", "zone", "RESISTANCE");
    }, signal);

    bindClick(elements.deleteAnchorButton, event => {
        preventEvent(event);

        const trade = getSelectedLearningTrade();

        if (!trade) {
            return;
        }

        trade.anchor = null;

        if (elements.learningAnchorTime) {
            elements.learningAnchorTime.value = "";
        }

        if (elements.learningAnchorPrice) {
            elements.learningAnchorPrice.value = "";
        }

        state.learning.pendingAnchorCenter = false;
        cancelAnchorPicker();
        renderLearningAnchor();
    }, signal);

    elements.learningRRModeButtons.forEach(button => {
        bindClick(button, event => {
            preventEvent(event);

            const trade = getSelectedLearningTrade();

            if (!trade) {
                return;
            }

            const mode = button.dataset.rrMode || "OFF";
            trade.rr_mode = mode;

            elements.learningRRModeButtons.forEach(item => {
                item.classList.toggle("active", item === button);
            });
        }, signal);
    });

    bindClick(elements.deleteTradeSetupButton, event => {
        preventEvent(event);

        const trade = getSelectedLearningTrade();

        if (!trade) {
            return;
        }

        trade.trade_geometry = createEmptyTradeGeometry();
        trade.breakout = null;
        trade.entry = null;
        trade.sl = null;
        trade.tp = null;

        elements.drawTradeSetupButton?.classList.remove("selected");
        callCallback("onClearTradeSetup", trade);
        notifyGeometryRefresh();
    }, signal);

    document.querySelectorAll(".learning-price-picker").forEach(button => {
        bindClick(button, event => {
            preventEvent(event);

            activatePricePicker(
                button.dataset.priceTarget,
                button
            );
        }, signal);
    });

    bindClick(elements.learningAnchorPicker, event => {
        preventEvent(event);
        activateAnchorPicker();
    }, signal);
}


// ============================================================================
// MODULE CLEANUP
// ============================================================================

export function destroyLearning() {
    callCallback("onCancelInteraction");
    cancelPricePicker();
    cancelAnchorPicker();

    if (listenerController) {
        listenerController.abort();
        listenerController = null;
    }

    initialized = false;
}

export function isLearningInitialized() {
    return initialized;
}
