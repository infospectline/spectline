
import { state } from "./state.js";
import {
    geometryTimeToX,
    geometryPriceToY,
    geometryXToTime,
    geometryYToPrice,
    snapXToCandleTime,
    getChartContainer,
    getMainPlotRect,
    isInsideMainPlot
} from "./chart.js";
import {
    ensureLearningTrade,
    getSelectedLearningTradeData,
    renumberLearningZones,
    updateLearningGeometrySummary
} from "./learning.js";


// ============================================================================
// MODULE STATE
// ============================================================================

const RENDERED_SELECTOR = '[data-geometry-renderer="learning"]';

const LEVELS = [
    {
        key: "breakout",
        label: "BREAKOUT",
        className: "learning-setup-breakout"
    },
    {
        key: "entry",
        label: "ENTRY",
        className: "learning-setup-entry"
    },
    {
        key: "sl",
        label: "SL",
        className: "learning-setup-sl"
    },
    {
        key: "tp",
        label: "TP",
        className: "learning-setup-tp"
    }
];

let overlayElement = null;
let previewElement = null;
let previewLevelNodes = [];
let previewFrame = null;
let renderFrame = null;

let callbacks = {
    onGeometryChange: null
};


// ============================================================================
// INTERNAL HELPERS
// ============================================================================

function getOverlay() {
    if (overlayElement?.isConnected) {
        return overlayElement;
    }

    overlayElement = document.getElementById("overlay");
    return overlayElement;
}

function getSelectedTrade() {
    return getSelectedLearningTradeData();
}

function isFiniteNumber(value) {
    return (
        value !== null &&
        value !== undefined &&
        value !== "" &&
        Number.isFinite(Number(value))
    );
}

function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function deepClone(value) {
    if (value === null || value === undefined) {
        return value;
    }

    return JSON.parse(JSON.stringify(value));
}

function formatPrice(price) {
    return isFiniteNumber(price)
        ? Number(price).toFixed(5)
        : "-";
}

function formatGeometryTime(time) {
    if (!isFiniteNumber(time)) {
        return "-";
    }

    const date = new Date(Number(time) * 1000);

    if (Number.isNaN(date.getTime())) {
        return "-";
    }

    return date.toISOString()
        .replace("T", " ")
        .replace(".000Z", " UTC");
}

function createEmptyTradeGeometry() {
    return {
        start_time: null,
        end_time: null,
        breakout: { time: null, price: null },
        entry: { time: null, price: null },
        sl: { time: null, price: null },
        tp: { time: null, price: null }
    };
}

function normalizeGeometryRect(rect) {
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

function syncSimpleTradeValues(trade) {
    if (!trade?.trade_geometry) {
        return;
    }

    trade.breakout = trade.trade_geometry.breakout?.price ?? null;
    trade.entry = trade.trade_geometry.entry?.price ?? null;
    trade.sl = trade.trade_geometry.sl?.price ?? null;
    trade.tp = trade.trade_geometry.tp?.price ?? null;
}

function ensureNextZoneId() {
    const highestId = state.learning.trades.reduce((maxId, trade) => {
        const tradeMax = (trade?.zones || []).reduce((zoneMax, zone) => {
            const id = Number(zone?.id);
            return Number.isFinite(id) ? Math.max(zoneMax, id) : zoneMax;
        }, 0);

        return Math.max(maxId, tradeMax);
    }, 0);

    state.learning.nextZoneId = Math.max(
        Number(state.learning.nextZoneId) || 1,
        highestId + 1
    );
}

function notifyGeometryChange(trade = getSelectedTrade()) {
    if (trade) {
        renumberLearningZones(trade);
    }

    renderLearningGeometry();
    updateLearningGeometrySummary();

    if (typeof callbacks.onGeometryChange === "function") {
        callbacks.onGeometryChange(trade || null);
    }
}

function markRendered(element) {
    element.dataset.geometryRenderer = "learning";
    return element;
}

function clearRenderedGeometry() {
    const overlay = getOverlay();

    if (!overlay) {
        return;
    }

    overlay.querySelectorAll(RENDERED_SELECTOR).forEach(element => {
        element.remove();
    });
}

function removePreview() {
    if (previewFrame !== null) {
        cancelAnimationFrame(previewFrame);
        previewFrame = null;
    }

    previewElement?.remove();
    previewElement = null;
    previewLevelNodes = [];
}

function clearDragVisualState() {
    const overlay = getOverlay();

    overlay?.querySelectorAll("[data-geometry-group]").forEach(element => {
        element.style.translate = "";
        element.style.transition = "";
    });

    if (typeof document !== "undefined" && document.body) {
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
    }
}

function restoreInteractionSnapshot(drag, trade) {
    if (!drag || !trade || !drag.original) {
        return;
    }

    if (Array.isArray(drag.original.zones)) {
        trade.zones = deepClone(drag.original.zones);
    }

    if (drag.original.trade !== undefined) {
        trade.trade_geometry = deepClone(drag.original.trade);
    }

    if (drag.original.simpleTradeValues) {
        Object.assign(
            trade,
            deepClone(drag.original.simpleTradeValues)
        );
    }
}

function scheduleGeometryRender() {
    if (renderFrame !== null) {
        return;
    }

    renderFrame = requestAnimationFrame(() => {
        renderFrame = null;
        renderLearningGeometry();
    });
}

function schedulePreviewRender() {
    if (previewFrame !== null) {
        return;
    }

    previewFrame = requestAnimationFrame(() => {
        previewFrame = null;
        renderGeometryPreview();
    });
}

function pointerEventToChartPoint(event, { allowOutside = false } = {}) {
    const container = getChartContainer();

    if (!container || !event) {
        return null;
    }

    const clientX = Number(event.clientX);
    const clientY = Number(event.clientY);

    if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) {
        return null;
    }

    if (
        !allowOutside &&
        window.matchMedia("(max-width: 900px) and (pointer: coarse)").matches &&
        !isInsideMainPlot(clientX, clientY)
    ) {
        return null;
    }

    const rect = container.getBoundingClientRect();
    const x = clientX - rect.left;
    const y = clientY - rect.top;

    if (
        !allowOutside &&
        (
            x < 0 ||
            x > rect.width ||
            y < 0 ||
            y > rect.height
        )
    ) {
        return null;
    }

    const time = geometryXToTime(x);
    const price = geometryYToPrice(y);

    if (!isFiniteNumber(time) || !isFiniteNumber(price)) {
        return null;
    }

    return {
        x,
        y,
        time: Number(time),
        price: Number(price),
        clientX,
        clientY
    };
}

function getGeometryObjectFromEvent(event) {
    if (!event) {
        return null;
    }

    const current = event.currentTarget;

    if (current?.dataset?.geometryObject) {
        return current;
    }

    const target = event.target instanceof Element
        ? event.target
        : null;

    return target?.closest(
        '[data-geometry-renderer="learning"][data-geometry-object]'
    ) || null;
}

function updateSetupDerivedFields(trade) {
    if (!trade?.trade_geometry) {
        return;
    }

    const stopLoss = Number(trade.trade_geometry.sl?.price);
    const takeProfit = Number(trade.trade_geometry.tp?.price);

    if (
        !Number.isFinite(stopLoss) ||
        !Number.isFinite(takeProfit) ||
        stopLoss === takeProfit
    ) {
        trade.direction = null;
    } else {
        trade.direction = takeProfit > stopLoss ? "LONG" : "SHORT";
    }

    syncSimpleTradeValues(trade);
}

function updateTradeLevelPrice(trade, level, price, originalGeometry) {
    if (
        !trade?.trade_geometry?.[level] ||
        !isFiniteNumber(price)
    ) {
        return;
    }

    trade.trade_geometry[level].price = Number(price);
    applyTradeRRMode(trade, level, originalGeometry);
    updateSetupDerivedFields(trade);
}


// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

export function initializeGeometry({
    overlay = null,
    onGeometryChange = null
} = {}) {
    if (renderFrame !== null) {
        cancelAnimationFrame(renderFrame);
        renderFrame = null;
    }

    removePreview();

    overlayElement = overlay || document.getElementById("overlay");

    callbacks.onGeometryChange = typeof onGeometryChange === "function"
        ? onGeometryChange
        : null;

    const container = getChartContainer();

    if (!container) {
        throw new Error('Chart container "#chart" was not found.');
    }

    state.interaction.activeTool = null;
    state.interaction.activeZoneType = null;
    state.interaction.activeDrag = null;

    ensureNextZoneId();

    container.classList.remove("learning-geometry-editing");

    return Boolean(overlayElement);
}


// ============================================================================
// GEOMETRY TOOL SELECTION
// ============================================================================

export function setGeometryTool(tool, zoneType = null) {
    const container = getChartContainer();

    if (tool !== null && tool !== "trade" && tool !== "zone") {
        return false;
    }

    if (
        tool === "zone" &&
        zoneType !== "SUPPORT" &&
        zoneType !== "RESISTANCE"
    ) {
        return false;
    }

    if (state.interaction.activeDrag) {
        cancelGeometryInteraction({
            restore: true,
            clearTool: false
        });
    }

    state.interaction.activeTool = tool;
    state.interaction.activeZoneType = tool === "zone" ? zoneType : null;

    container?.classList.remove("learning-geometry-editing");

    if (tool) {
        container?.classList.add("learning-geometry-editing");
    }

    if (tool !== "trade") {
        document.getElementById("drawTradeSetupButton")
            ?.classList.remove("selected");
    }

    return true;
}

export function getGeometryTool() {
    return state.interaction.activeTool;
}

export function getActiveGeometryDrag() {
    return state.interaction.activeDrag;
}

export function isGeometryInteractionActive() {
    return state.interaction.activeDrag !== null;
}


// ============================================================================
// GEOMETRY RENDERING
// ============================================================================

function drawGeometryRect(
    rect,
    className,
    label,
    objectType,
    zoneId = null,
    zoneColor = null
) {
    const normalized = normalizeGeometryRect(rect);

    if (!normalized) {
        return false;
    }

    const x1 = geometryTimeToX(normalized.start_time);
    const x2 = geometryTimeToX(normalized.end_time);
    const y1 = geometryPriceToY(normalized.high);
    const y2 = geometryPriceToY(normalized.low);

    if (![x1, x2, y1, y2].every(isFiniteNumber)) {
        return false;
    }

    const left = Math.min(x1, x2);
    const right = Math.max(x1, x2);
    const top = Math.min(y1, y2);
    const bottom = Math.max(y1, y2);

    const width = Math.max(8, right - left);
    const height = Math.max(8, bottom - top);

    const rectangle = markRendered(document.createElement("div"));

    rectangle.className = `learning-geometry-rect ${className}`;

    const geometryGroup = objectType === "zone"
        ? `zone:${zoneId}`
        : objectType;

    rectangle.dataset.geometryGroup = geometryGroup;
    rectangle.dataset.geometryObject = objectType;

    if (zoneId !== null) {
        rectangle.dataset.geometryZoneId = String(zoneId);
    }

    if (zoneColor) {
        rectangle.style.setProperty("--zone-color", zoneColor);
        rectangle.style.setProperty("--zone-bg", `${zoneColor}24`);
        rectangle.style.setProperty("--zone-glow", `${zoneColor}55`);
    }

    rectangle.style.left = `${left}px`;
    rectangle.style.top = `${top}px`;
    rectangle.style.width = `${width}px`;
    rectangle.style.height = `${height}px`;
    rectangle.title = `Drag ${label}`;

    getOverlay()?.appendChild(rectangle);

    const labelElement = markRendered(document.createElement("div"));

    labelElement.className = "learning-setup-label";
    labelElement.dataset.geometryGroup = geometryGroup;
    labelElement.textContent = label;
    labelElement.style.left = `${left + 6}px`;
    labelElement.style.top = `${top + 6}px`;
    labelElement.style.pointerEvents = "none";

    getOverlay()?.appendChild(labelElement);

    createGeometryResizeHandles(
        objectType,
        left,
        top,
        width,
        height,
        geometryGroup,
        zoneId
    );

    return true;
}

function createGeometryResizeHandles(
    objectType,
    left,
    top,
    width,
    height,
    geometryGroup = null,
    zoneId = null
) {
    const handles = [
        { edge: "left", left: left - 4, top, width: 8, height },
        { edge: "right", left: left + width - 4, top, width: 8, height },
        { edge: "top", left, top: top - 4, width, height: 8 },
        { edge: "bottom", left, top: top + height - 4, width, height: 8 }
    ];

    const overlay = getOverlay();

    if (!overlay) {
        return;
    }

    handles.forEach(info => {
        const handle = markRendered(document.createElement("div"));

        handle.className = `learning-geometry-resize-handle ${info.edge}`;
        handle.dataset.geometryObject = objectType;
        handle.dataset.geometryEdge = info.edge;

        if (zoneId !== null) {
            handle.dataset.geometryZoneId = String(zoneId);
        }

        if (geometryGroup) {
            handle.dataset.geometryGroup = geometryGroup;
        }

        handle.style.left = `${info.left}px`;
        handle.style.top = `${info.top}px`;
        handle.style.width = `${info.width}px`;
        handle.style.height = `${info.height}px`;

        overlay.appendChild(handle);
    });
}

function addSetupArea(left, width, entryY, targetPrice, className) {
    if (
        entryY === null ||
        targetPrice === null ||
        targetPrice === undefined
    ) {
        return;
    }

    const targetY = geometryPriceToY(targetPrice);

    if (!isFiniteNumber(targetY)) {
        return;
    }

    const area = markRendered(document.createElement("div"));

    area.className = `learning-setup-area ${className}`;
    area.dataset.geometryGroup = "trade";
    area.style.left = `${left}px`;
    area.style.top = `${Math.min(entryY, targetY)}px`;
    area.style.width = `${width}px`;
    area.style.height = `${Math.max(1, Math.abs(targetY - entryY))}px`;

    getOverlay()?.appendChild(area);
}

function createTradeSetupEdge(edge, top, height, visualLeft) {
    const handle = markRendered(document.createElement("div"));

    handle.className = `learning-trade-setup-edge ${edge}`;
    handle.dataset.geometryObject = "trade";
    handle.dataset.geometryGroup = "trade";
    handle.dataset.geometryEdge = edge;

    handle.style.left = `${visualLeft - 3}px`;
    handle.style.top = `${top}px`;
    handle.style.height = `${height}px`;

    getOverlay()?.appendChild(handle);
}

function drawTradeSetupGeometry(geometry) {
    if (
        !geometry ||
        geometry.start_time === null ||
        geometry.start_time === undefined ||
        geometry.end_time === null ||
        geometry.end_time === undefined
    ) {
        return false;
    }

    const x1 = geometryTimeToX(geometry.start_time);
    const x2 = geometryTimeToX(geometry.end_time);

    if (!isFiniteNumber(x1) || !isFiniteNumber(x2)) {
        return false;
    }

    const prices = LEVELS
        .map(level => geometry[level.key]?.price)
        .filter(isFiniteNumber)
        .map(Number);

    if (!prices.length) {
        return false;
    }

    const high = Math.max(...prices);
    const low = Math.min(...prices);

    const y1 = geometryPriceToY(high);
    const y2 = geometryPriceToY(low);

    if (!isFiniteNumber(y1) || !isFiniteNumber(y2)) {
        return false;
    }

    const left = Math.min(x1, x2);
    const right = Math.max(x1, x2);
    const top = Math.min(y1, y2);
    const bottom = Math.max(y1, y2);

    const width = Math.max(12, right - left);
    const height = Math.max(20, bottom - top);

    const rectangle = markRendered(document.createElement("div"));

    rectangle.className =
        "learning-geometry-rect learning-trade-setup-rect";

    rectangle.dataset.geometryObject = "trade";
    rectangle.dataset.geometryGroup = "trade";

    rectangle.style.left = `${left}px`;
    rectangle.style.top = `${top}px`;
    rectangle.style.width = `${width}px`;
    rectangle.style.height = `${height}px`;

    rectangle.title = "Drag whole TRADE SETUP";

    getOverlay()?.appendChild(rectangle);

    const entryY = isFiniteNumber(geometry.entry?.price)
        ? geometryPriceToY(geometry.entry.price)
        : null;

    addSetupArea(
        left,
        width,
        entryY,
        geometry.tp?.price,
        "learning-setup-profit-area"
    );

    addSetupArea(
        left,
        width,
        entryY,
        geometry.sl?.price,
        "learning-setup-risk-area"
    );

    createTradeSetupEdge("left", top, height, left);
    createTradeSetupEdge("right", top, height, right);

    LEVELS.forEach(level => {
        const point = geometry[level.key];

        if (!point || !isFiniteNumber(point.price)) {
            return;
        }

        const y = geometryPriceToY(point.price);

        if (!isFiniteNumber(y)) {
            return;
        }

        const line = markRendered(document.createElement("div"));

        line.className = `learning-setup-level ${level.className}`;
        line.dataset.geometryLevel = level.key;
        line.dataset.geometryObject = "trade";
        line.dataset.geometryGroup = "trade";

        line.style.left = `${left}px`;
        line.style.width = `${width}px`;
        line.style.top = `${y - 2}px`;

        line.title = `Drag ${level.label}`;

        getOverlay()?.appendChild(line);

        const labelElement = markRendered(document.createElement("div"));

        labelElement.className = "learning-setup-label";
        labelElement.dataset.geometryGroup = "trade";

        labelElement.textContent =
            `${level.label} ${formatPrice(point.price)} | ${formatGeometryTime(point.time)}`;

        labelElement.style.left = `${left + 6}px`;
        labelElement.style.top = `${y - 18}px`;
        labelElement.style.pointerEvents = "none";

        getOverlay()?.appendChild(labelElement);
    });

    return true;
}

export function renderLearningGeometry() {
    const overlay = getOverlay();
    const trade = getSelectedTrade();
    const detailView = document.getElementById("learningDetailView");

    clearRenderedGeometry();
    syncGeometryButtonStates();

    if (
        !overlay ||
        !state.learning.mode ||
        !trade ||
        (detailView && detailView.style.display === "none")
    ) {
        return false;
    }

    ensureLearningTrade(trade);
    renumberLearningZones(trade);

    trade.zones.forEach(zone => {
        drawGeometryRect(
            zone.geometry,
            "learning-zone-rect",
            `${zone.type} ${zone.number}`,
            "zone",
            zone.id,
            zone.color
        );
    });

    drawTradeSetupGeometry(trade.trade_geometry);

    return true;
}

export function syncGeometryButtonStates() {
    const trade = getSelectedTrade();
    const button = document.getElementById("drawTradeSetupButton");

    const hasTradeSetup = Boolean(
        trade?.trade_geometry &&
        trade.trade_geometry.start_time != null &&
        trade.trade_geometry.end_time != null
    );

    button?.classList.toggle("active", hasTradeSetup);
}


// ============================================================================
// DRAWING OPERATIONS
// ============================================================================

export function beginGeometryDraw(event) {
    if (
        !state.interaction.activeTool ||
        state.interaction.activeDrag
    ) {
        return false;
    }

    const trade = getSelectedTrade();

    if (!trade) {
        return false;
    }

    const point = pointerEventToChartPoint(event);

    if (!point) {
        return false;
    }

    ensureLearningTrade(trade);

    state.interaction.activeDrag = {
        mode: "draw",
        tool: state.interaction.activeTool,
        zoneType: state.interaction.activeZoneType,
        zoneColor: state.interaction.activeTool === "zone"
            ? (
                state.interaction.activeZoneType === "SUPPORT"
                    ? "#00d26a"
                    : "#ff3b30"
            )
            : null,
        start: point,
        current: point,
        lastValidPoint: point
    };

    schedulePreviewRender();

    return true;
}

export function updateGeometryDraw(event) {
    const drag = state.interaction.activeDrag;

    if (!drag || drag.mode !== "draw") {
        return false;
    }

    const point = pointerEventToChartPoint(event);

    if (!point) {
        return false;
    }

    drag.current = point;
    drag.lastValidPoint = point;

    schedulePreviewRender();

    return true;
}

function renderGeometryPreview() {
    const drag = state.interaction.activeDrag;
    const overlay = getOverlay();

    if (!drag || drag.mode !== "draw" || !overlay) {
        removePreview();
        return;
    }

    const start = drag.start;
    const current = drag.current || drag.lastValidPoint || start;

    const startX = geometryTimeToX(start.time);
    const currentX = geometryTimeToX(current.time);

    const startY = geometryPriceToY(start.price);
    const currentY = geometryPriceToY(current.price);

    if (![startX, currentX, startY, currentY].every(isFiniteNumber)) {
        return;
    }

    const left = Math.min(startX, currentX);
    const top = Math.min(startY, currentY);

    const width = Math.max(2, Math.abs(currentX - startX));
    const height = Math.max(2, Math.abs(currentY - startY));

    if (!previewElement) {
        previewElement = document.createElement("div");

        previewElement.className =
            "learning-geometry-rect learning-geometry-preview";

        previewElement.style.pointerEvents = "none";
        previewElement.style.position = "absolute";
        previewElement.style.boxSizing = "border-box";

        overlay.appendChild(previewElement);
    }

    previewElement.style.left = `${left}px`;
    previewElement.style.top = `${top}px`;
    previewElement.style.width = `${width}px`;
    previewElement.style.height = `${height}px`;

    if (drag.tool === "zone") {
        if (previewElement.dataset.previewTool !== "zone") {
            previewElement.replaceChildren();
            previewLevelNodes = [];
            previewElement.dataset.previewTool = "zone";
        }

        const color = drag.zoneColor || "#a456ff";

        previewElement.style.background = `${color}24`;
        previewElement.style.border = `1px dashed ${color}`;

        previewElement.style.setProperty("--zone-color", color);
        previewElement.style.setProperty("--zone-bg", `${color}24`);

        return;
    }

    previewElement.style.background = "rgba(42, 156, 255, 0.06)";
    previewElement.style.border = "1px dashed #2a9cff";

    if (previewElement.dataset.previewTool !== "trade") {
        previewElement.replaceChildren();

        previewLevelNodes = LEVELS.map(level => {
            const line = document.createElement("div");

            line.className =
                `learning-setup-level ${level.className} learning-geometry-preview-level`;

            line.style.position = "absolute";
            line.style.left = "0";
            line.style.width = "100%";
            line.style.pointerEvents = "none";

            const label = document.createElement("div");

            label.className =
                "learning-setup-label learning-geometry-preview-label";

            label.style.position = "absolute";
            label.style.left = "6px";
            label.style.pointerEvents = "none";

            previewElement.append(line, label);

            return { line, label };
        });

        previewElement.dataset.previewTool = "trade";
    }

    const priceRange = Number(current.price) - Number(start.price);
    const directionSign = priceRange >= 0 ? 1 : -1;
    const absoluteRange = Math.abs(priceRange);

    const previewPrices = [
        Number(start.price) + directionSign * absoluteRange * 0.33,
        Number(start.price) + directionSign * absoluteRange * 0.66,
        Number(start.price),
        Number(current.price)
    ];

    const previewTimes = [
        start.time + (current.time - start.time) * 0.33,
        start.time + (current.time - start.time) * 0.66,
        start.time,
        current.time
    ];

    LEVELS.forEach((level, index) => {
        const absoluteY = geometryPriceToY(previewPrices[index]);
        const nodes = previewLevelNodes[index];

        if (!nodes || !isFiniteNumber(absoluteY)) {
            return;
        }

        nodes.line.style.top = `${absoluteY - top}px`;
        nodes.label.style.top = `${absoluteY - top - 18}px`;

        nodes.label.textContent =
            `${level.label} ${formatPrice(previewPrices[index])} | ${formatGeometryTime(previewTimes[index])}`;
    });
}

function syncTradeDirectionFromSetup(trade) {
    if (!trade?.trade_geometry) {
        return;
    }

    const stopLoss = Number(trade.trade_geometry.sl?.price);
    const takeProfit = Number(trade.trade_geometry.tp?.price);

    if (
        !Number.isFinite(stopLoss) ||
        !Number.isFinite(takeProfit) ||
        stopLoss === takeProfit
    ) {
        trade.direction = null;
        return;
    }

    trade.direction = takeProfit > stopLoss ? "LONG" : "SHORT";
}

export function finishGeometryDraw(event) {
    const drag = state.interaction.activeDrag;

    if (!drag || drag.mode !== "draw") {
        return false;
    }

    let completed = false;

    try {
        const finalPoint = pointerEventToChartPoint(event);

        if (finalPoint) {
            drag.current = finalPoint;
            drag.lastValidPoint = finalPoint;
        }

        const start = drag.start;
        const end = drag.current || drag.lastValidPoint || start;
        const trade = getSelectedTrade();

        if (!trade || !start || !end) {
            return false;
        }

        ensureLearningTrade(trade);

        const startTime = Math.min(start.time, end.time);
        const endTime = Math.max(start.time, end.time);
        const low = Math.min(start.price, end.price);
        const high = Math.max(start.price, end.price);

        if (drag.tool === "zone") {
            if (
                drag.zoneType !== "SUPPORT" &&
                drag.zoneType !== "RESISTANCE"
            ) {
                return false;
            }

            ensureNextZoneId();

            trade.zones.push({
                id: state.learning.nextZoneId++,
                type: drag.zoneType,
                number: 0,
                color: drag.zoneColor,
                geometry: {
                    start_time: startTime,
                    end_time: endTime,
                    low,
                    high
                }
            });

            renumberLearningZones(trade);
            completed = true;

        } else if (drag.tool === "trade") {
            const startPrice = Number(start.price);
            const endPrice = Number(end.price);

            const range = Math.abs(endPrice - startPrice);

            if (!Number.isFinite(range) || range <= 0) {
                return false;
            }

            const directionSign = endPrice > startPrice ? 1 : -1;
            const timeRange = endTime - startTime;

            trade.trade_geometry = {
                start_time: startTime,
                end_time: endTime,

                breakout: {
                    time: startTime + timeRange * 0.33,
                    price: startPrice + directionSign * range * 0.33
                },

                entry: {
                    time: startTime + timeRange * 0.66,
                    price: startPrice + directionSign * range * 0.66
                },

                sl: {
                    time: startTime,
                    price: startPrice
                },

                tp: {
                    time: endTime,
                    price: endPrice
                }
            };

            syncTradeDirectionFromSetup(trade);
            syncSimpleTradeValues(trade);

            completed = true;
        }

        return completed;

    } catch (error) {
        console.error("Failed to finish geometry drawing:", error);
        return false;

    } finally {
        state.interaction.activeDrag = null;

        removePreview();
        setGeometryTool(null);

        if (completed) {
            notifyGeometryChange();
        } else {
            renderLearningGeometry();
        }
    }
}


// ============================================================================
// GEOMETRY EDITING OPERATIONS
// ============================================================================

export function beginGeometryObjectInteraction(event) {
    if (
        state.interaction.activeDrag ||
        state.interaction.activeTool ||
        (event?.button !== undefined && event.button !== 0)
    ) {
        return false;
    }

    const trade = getSelectedTrade();

    if (!trade) {
        return false;
    }

    const object = getGeometryObjectFromEvent(event);

    if (!object) {
        return false;
    }

    const objectType = object.dataset.geometryObject;
    const edge = object.dataset.geometryEdge || null;
    const level = object.dataset.geometryLevel || null;
    const zoneId = object.dataset.geometryZoneId || null;
    const geometryGroup = object.dataset.geometryGroup || null;

    const point = pointerEventToChartPoint(event);

    if (!point || !objectType) {
        return false;
    }

    ensureLearningTrade(trade);

    state.interaction.activeDrag = {
        mode: level
            ? "level"
            : edge
                ? (
                    objectType === "trade"
                        ? "setup-resize"
                        : "resize"
                )
                : "move",

        objectType,
        zoneId,
        edge,
        level,
        geometryGroup,

        start: point,
        current: point,
        lastValidPoint: point,

        startClientX: Number(event.clientX),
        startClientY: Number(event.clientY),

        original: {
            zones: deepClone(trade.zones || []),
            trade: deepClone(trade.trade_geometry),

            simpleTradeValues: {
                direction: trade.direction ?? null,
                breakout: trade.breakout ?? null,
                entry: trade.entry ?? null,
                sl: trade.sl ?? null,
                tp: trade.tp ?? null
            }
        }
    };

    if (state.interaction.activeDrag.mode === "move") {
        document.body.style.userSelect = "none";
        document.body.style.cursor = "grabbing";

        const overlay = getOverlay();

        overlay?.querySelectorAll("[data-geometry-group]").forEach(element => {
            if (element.dataset.geometryGroup === geometryGroup) {
                element.style.transition = "none";
            }
        });
    }

    return true;
}

function applyTradeRRMode(trade, changedLevel, originalGeometry = null) {
    const mode = trade?.rr_mode || "OFF";

    if (mode === "OFF") {
        return;
    }

    const geometry = trade.trade_geometry;

    if (!isFiniteNumber(geometry?.entry?.price)) {
        return;
    }

    const entry = Number(geometry.entry.price);

    const rr = mode === "2x"
        ? 2
        : mode === "3x"
            ? 3
            : null;

    if (rr === null || !Number.isFinite(entry)) {
        return;
    }

    if (changedLevel === "entry") {
        if (
            originalGeometry?.entry?.price == null ||
            originalGeometry?.sl?.price == null ||
            originalGeometry?.tp?.price == null
        ) {
            return;
        }

        const originalEntry = Number(originalGeometry.entry.price);
        const originalSL = Number(originalGeometry.sl.price);
        const originalTP = Number(originalGeometry.tp.price);

        if (![originalEntry, originalSL, originalTP].every(Number.isFinite)) {
            return;
        }

        const movement = entry - originalEntry;

        if (movement === 0) {
            return;
        }

        const movedTowardSL =
            movement * (originalSL - originalEntry) > 0;

        const movedTowardTP =
            movement * (originalTP - originalEntry) > 0;

        if (movedTowardSL) {
            const risk = Math.abs(entry - originalSL);

            if (risk <= 0) {
                return;
            }

            const direction = entry > originalSL ? 1 : -1;

            geometry.sl.price = originalSL;
            geometry.tp.price = entry + direction * risk * rr;

            return;
        }

        if (movedTowardTP) {
            const reward = Math.abs(originalTP - entry);

            if (reward <= 0) {
                return;
            }

            const direction = originalTP > entry ? 1 : -1;

            geometry.tp.price = originalTP;
            geometry.sl.price = entry - direction * (reward / rr);

            return;
        }
    }

    if (
        changedLevel === "sl" &&
        isFiniteNumber(geometry.sl?.price)
    ) {
        const sl = Number(geometry.sl.price);
        const risk = Math.abs(entry - sl);

        if (risk <= 0) {
            return;
        }

        const direction = entry > sl ? 1 : -1;

        geometry.tp.price = entry + direction * risk * rr;
    }

    if (
        changedLevel === "tp" &&
        isFiniteNumber(geometry.tp?.price)
    ) {
        const tp = Number(geometry.tp.price);
        const reward = Math.abs(tp - entry);

        if (reward <= 0) {
            return;
        }

        const direction = tp > entry ? 1 : -1;

        geometry.sl.price = entry - direction * (reward / rr);
    }
}

function translateGeometryGroup(group, dx, dy) {
    if (!group) {
        return;
    }

    const overlay = getOverlay();

    overlay?.querySelectorAll("[data-geometry-group]").forEach(element => {
        if (element.dataset.geometryGroup === group) {
            element.style.translate = `${dx}px ${dy}px`;
        }
    });
}

function resizeZoneGeometry(trade, zoneId, edge, original, current) {
    const source = original.zones.find(
        zone => String(zone.id) === String(zoneId)
    );

    if (!source?.geometry) {
        return false;
    }

    const result = { ...source.geometry };

    if (edge === "left") {
        result.start_time = Math.min(
            current.time,
            Number(source.geometry.end_time) - 1
        );

    } else if (edge === "right") {
        result.end_time = Math.max(
            current.time,
            Number(source.geometry.start_time) + 1
        );

    } else if (edge === "top") {
        result.high = Math.max(
            current.price,
            Number(source.geometry.low)
        );

    } else if (edge === "bottom") {
        result.low = Math.min(
            current.price,
            Number(source.geometry.high)
        );

    } else {
        return false;
    }

    const target = trade.zones.find(
        zone => String(zone.id) === String(zoneId)
    );

    if (!target) {
        return false;
    }

    target.geometry = result;

    return true;
}

function resizeTradeSetupHorizontally(trade, edge, original, current) {
    if (!trade || !original) {
        return false;
    }

    const oldStart = Number(original.start_time);
    const oldEnd = Number(original.end_time);
    const oldSpan = oldEnd - oldStart;

    if (!Number.isFinite(oldSpan) || oldSpan <= 0) {
        return false;
    }

    let newStart = oldStart;
    let newEnd = oldEnd;

    if (edge === "left") {
        newStart = Math.min(
            Number(current.time),
            oldEnd - 1
        );

    } else if (edge === "right") {
        newEnd = Math.max(
            Number(current.time),
            oldStart + 1
        );

    } else {
        return false;
    }

    trade.trade_geometry.start_time = newStart;
    trade.trade_geometry.end_time = newEnd;

    const newSpan = newEnd - newStart;

    LEVELS.forEach(level => {
        const point = original[level.key];

        if (!point) {
            return;
        }

        const ratio = clamp(
            (Number(point.time) - oldStart) / oldSpan,
            0,
            1
        );

        trade.trade_geometry[level.key] = {
            time: newStart + newSpan * ratio,
            price: point.price
        };
    });

    syncSimpleTradeValues(trade);

    return true;
}


function getSnappedCandleRange(originalStartTime, originalEndTime, deltaX) {
    const candles = state.chart.lastData?.h1;

    if (
        !Array.isArray(candles) ||
        !candles.length ||
        !isFiniteNumber(originalStartTime) ||
        !isFiniteNumber(originalEndTime) ||
        !isFiniteNumber(deltaX)
    ) {
        return null;
    }

    const findNearestIndex = timestamp => {
        let bestIndex = -1;
        let bestDistance = Infinity;

        for (let index = 0; index < candles.length; index += 1) {
            const candleTime = Number(candles[index]?.time);

            if (!Number.isFinite(candleTime)) {
                continue;
            }

            const distance = Math.abs(candleTime - Number(timestamp));

            if (distance < bestDistance) {
                bestDistance = distance;
                bestIndex = index;
            }
        }

        return bestIndex;
    };

    const originalStartIndex = findNearestIndex(originalStartTime);
    const originalEndIndex = findNearestIndex(originalEndTime);

    if (
        originalStartIndex < 0 ||
        originalEndIndex < originalStartIndex
    ) {
        return null;
    }

    // Preserve the original width in candle intervals.
    const spanBars = originalEndIndex - originalStartIndex;

    const originalStartX = geometryTimeToX(originalStartTime);

    if (!isFiniteNumber(originalStartX)) {
        return null;
    }

    // Find the candle nearest to the requested drop position.
    const targetTime = snapXToCandleTime(
        Number(originalStartX) + Number(deltaX),
        candles
    );

    if (!isFiniteNumber(targetTime)) {
        return null;
    }

    const requestedIndex = findNearestIndex(targetTime);
    const maxStartIndex = candles.length - 1 - spanBars;

    if (requestedIndex < 0 || maxStartIndex < 0) {
        return null;
    }

    // Keep the entire object inside the available candle history.
    const targetStartIndex = clamp(
        requestedIndex,
        0,
        maxStartIndex
    );

    const startTime = Number(candles[targetStartIndex]?.time);
    const endTime = Number(candles[targetStartIndex + spanBars]?.time);

    const startX = geometryTimeToX(startTime);
    const endX = geometryTimeToX(endTime);

    if (
        !isFiniteNumber(startTime) ||
        !isFiniteNumber(endTime) ||
        !isFiniteNumber(startX) ||
        !isFiniteNumber(endX) ||
        endTime < startTime
    ) {
        return null;
    }

    return {
        startTime,
        endTime
    };
}

function commitFreeMoveInteraction(drag, event) {
    const trade = getSelectedTrade();

    if (!trade || !drag?.original) {
        return false;
    }

    const container = getChartContainer();
    const containerRect = container?.getBoundingClientRect();
    const plotRect = getMainPlotRect();

    if (!container || !containerRect || !plotRect) {
        return false;
    }

    const releaseClientX = isFiniteNumber(event?.clientX)
        ? Number(event.clientX)
        : Number(drag.current?.clientX ?? drag.startClientX);

    const rawReleaseClientY = isFiniteNumber(event?.clientY)
        ? Number(event.clientY)
        : Number(drag.current?.clientY ?? drag.startClientY);

    const clampedClientY = clamp(
        rawReleaseClientY,
        plotRect.top,
        Math.max(plotRect.top, plotRect.bottom - 1)
    );

    const releaseY = clampedClientY - containerRect.top;
    const releasePrice = geometryYToPrice(releaseY);
    const lastPrice = drag.current?.price ?? drag.start.price;

    const deltaPrice = isFiniteNumber(releasePrice)
        ? Number(releasePrice) - Number(drag.start.price)
        : Number(lastPrice) - Number(drag.start.price);

    const deltaX = releaseClientX - Number(drag.startClientX);

    if (!Number.isFinite(deltaPrice) || !Number.isFinite(deltaX)) {
        return false;
    }

    if (
        drag.objectType === "zone" &&
        drag.zoneId !== null
    ) {
        const source = drag.original.zones.find(
            zone => String(zone.id) === String(drag.zoneId)
        );

        const target = trade.zones.find(
            zone => String(zone.id) === String(drag.zoneId)
        );

        if (!source?.geometry || !target) {
            return false;
        }
    
        const snappedRange = getSnappedCandleRange(
            source.geometry.start_time,
            source.geometry.end_time,
            deltaX
        );

        if (!snappedRange) {
            return false;
        }

        const low = Number(source.geometry.low) + deltaPrice;
        const high = Number(source.geometry.high) + deltaPrice;

        const x1 = geometryTimeToX(snappedRange.startTime);
        const x2 = geometryTimeToX(snappedRange.endTime);
        const y1 = geometryPriceToY(high);
        const y2 = geometryPriceToY(low);

        if (
            ![low, high, x1, x2, y1, y2].every(isFiniteNumber)
        ) {
            return false;
        }

        target.geometry = {
            ...source.geometry,
            start_time: snappedRange.startTime,
            end_time: snappedRange.endTime,
            low,
            high
        };

        return true;

    }

    if (
        drag.objectType === "trade" &&
        drag.original.trade
    ) {
        
    const source = drag.original.trade;

    const snappedRange = getSnappedCandleRange(
        source.start_time,
        source.end_time,
        deltaX
    );

    if (!snappedRange) {
        return false;
    }

    const oldStart = Number(source.start_time);
    const oldEnd = Number(source.end_time);

    const oldSpan = oldEnd - oldStart;
    const newSpan = snappedRange.endTime - snappedRange.startTime;

    const nextGeometry = deepClone(source);

    nextGeometry.start_time = snappedRange.startTime;
    nextGeometry.end_time = snappedRange.endTime;

    LEVELS.forEach(level => {
        const sourcePoint = source[level.key];

        if (!sourcePoint) {
            return;
        }

        const pointTime = Number(sourcePoint.time);

        const ratio = (
            oldSpan > 0 &&
            Number.isFinite(pointTime)
        )
            ? clamp(
                (pointTime - oldStart) / oldSpan,
                0,
                1
            )
            : 0;

        nextGeometry[level.key] = {
            ...sourcePoint,

            time: isFiniteNumber(sourcePoint.time)
                ? snappedRange.startTime + newSpan * ratio
                : sourcePoint.time,

            price: isFiniteNumber(sourcePoint.price)
                ? Number(sourcePoint.price) + deltaPrice
                : sourcePoint.price
        };
    });

    const prices = LEVELS
        .map(level => nextGeometry[level.key]?.price)
        .filter(isFiniteNumber)
        .map(Number);

    if (!prices.length) {
        return false;
    }

    const x1 = geometryTimeToX(nextGeometry.start_time);
    const x2 = geometryTimeToX(nextGeometry.end_time);

    const y1 = geometryPriceToY(Math.max(...prices));
    const y2 = geometryPriceToY(Math.min(...prices));

    if (![x1, x2, y1, y2].every(isFiniteNumber)) {
        return false;
    }

    ensureLearningTrade(trade);

    trade.trade_geometry = nextGeometry;

    updateSetupDerivedFields(trade);

    return true;

    }

    return false;
}

function applyCurrentObjectEdit(drag, event) {
    const trade = getSelectedTrade();

    if (!trade || !drag) {
        return false;
    }

    const current = pointerEventToChartPoint(event);

    if (!current) {
        return false;
    }

    drag.current = current;
    drag.lastValidPoint = current;

    const original = drag.original;
    let changed = false;

    if (
        drag.objectType === "zone" &&
        drag.edge !== null
    ) {
        changed = resizeZoneGeometry(
            trade,
            drag.zoneId,
            drag.edge,
            original,
            current
        ) || changed;
    }

    if (drag.mode === "setup-resize") {
        changed = resizeTradeSetupHorizontally(
            trade,
            drag.edge,
            original.trade,
            current
        ) || changed;
    }

    if (drag.mode === "level") {
        const level = drag.level;
        const price = geometryYToPrice(current.y);

        if (
            level &&
            isFiniteNumber(price) &&
            trade.trade_geometry?.[level]
        ) {
            updateTradeLevelPrice(
                trade,
                level,
                price,
                original.trade
            );

            changed = true;
        }
    }

    if (changed) {
        scheduleGeometryRender();
    }

    return changed;
}

export function updateGeometryObjectInteraction(event) {
    const drag = state.interaction.activeDrag;

    if (!drag || drag.mode === "draw") {
        return false;
    }

    if (drag.mode === "move") {
        const current = pointerEventToChartPoint(event);

        if (current) {
            drag.current = current;
            drag.lastValidPoint = current;
        }

        const clientX = Number(event?.clientX);
        const clientY = Number(event?.clientY);

        if (!Number.isFinite(clientX) || !Number.isFinite(clientY)) {
            return false;
        }

        const dx = clientX - Number(drag.startClientX);
        const dy = clientY - Number(drag.startClientY);

        translateGeometryGroup(
            drag.geometryGroup,
            dx,
            dy
        );

        return true;
    }

    return applyCurrentObjectEdit(drag, event);
}

export function finishGeometryObjectInteraction(event) {
    const drag = state.interaction.activeDrag;

    if (!drag || drag.mode === "draw") {
        return false;
    }

    let committed = true;

    try {
        if (drag.mode === "move") {
            committed = commitFreeMoveInteraction(drag, event);

            if (!committed) {
                const trade = getSelectedTrade();
                restoreInteractionSnapshot(drag, trade);
            }

        } else {
            const finalPoint = pointerEventToChartPoint(event);

            if (finalPoint) {
                applyCurrentObjectEdit(drag, event);
            }

            const trade = getSelectedTrade();

            if (trade) {
                updateSetupDerivedFields(trade);
            }
        }

        return committed;

    } catch (error) {
        console.error("Failed to finish geometry interaction:", error);

        const trade = getSelectedTrade();
        restoreInteractionSnapshot(drag, trade);

        return false;

    } finally {
        state.interaction.activeDrag = null;

        clearDragVisualState();
        renderLearningGeometry();
        updateLearningGeometrySummary();
    }
}

export function cancelGeometryInteraction({
    restore = true,
    clearTool = false
} = {}) {
    const drag = state.interaction.activeDrag;

    if (drag && restore && drag.mode !== "draw") {
        restoreInteractionSnapshot(drag, getSelectedTrade());
    }

    state.interaction.activeDrag = null;

    removePreview();
    clearDragVisualState();

    if (clearTool) {
        setGeometryTool(null);
    }

    renderLearningGeometry();
    updateLearningGeometrySummary();

    return true;
}


// ============================================================================
// GEOMETRY HIT TESTING
// ============================================================================

export function findEditableGeometryAtPoint(clientX, clientY) {
    const overlay = getOverlay();

    if (
        !overlay ||
        !isFiniteNumber(clientX) ||
        !isFiniteNumber(clientY)
    ) {
        return null;
    }

    const hitRadius = 18;

    let bestHandle = null;
    let bestHandleScore = Infinity;

    const handles = overlay.querySelectorAll(
        ".learning-setup-level, " +
        ".learning-trade-setup-edge, " +
        ".learning-geometry-resize-handle"
    );

    for (const element of handles) {
        if (
            element.classList.contains("manual-setup-view-only") ||
            element.dataset.geometryRenderer !== "learning"
        ) {
            continue;
        }

        const rect = element.getBoundingClientRect();

        let distance;
        let priority = 0;

        if (element.classList.contains("learning-setup-level")) {
            if (
                clientX < rect.left - 8 ||
                clientX > rect.right + 8
            ) {
                continue;
            }

            distance = Math.abs(
                clientY - (rect.top + rect.bottom) / 2
            );

        } else {
            const dx = clientX < rect.left
                ? rect.left - clientX
                : clientX > rect.right
                    ? clientX - rect.right
                    : 0;

            const dy = clientY < rect.top
                ? rect.top - clientY
                : clientY > rect.bottom
                    ? clientY - rect.bottom
                    : 0;

            distance = Math.hypot(dx, dy);

            priority = element.classList.contains(
                "learning-trade-setup-edge"
            ) ? 1 : 2;
        }

        if (distance > hitRadius) {
            continue;
        }

        const score = distance + priority;

        if (score < bestHandleScore) {
            bestHandle = element;
            bestHandleScore = score;
        }
    }

    if (bestHandle) {
        return bestHandle;
    }

    let bestZone = null;
    let bestZoneZ = -Infinity;

    let bestOther = null;
    let bestOtherZ = -Infinity;

    const bodies = overlay.querySelectorAll(
        ".learning-zone-rect, .learning-geometry-rect"
    );

    for (const element of bodies) {
        if (
            element.classList.contains("manual-setup-view-only") ||
            element.dataset.geometryRenderer !== "learning" ||
            !element.dataset.geometryObject
        ) {
            continue;
        }

        const rect = element.getBoundingClientRect();

        if (
            clientX < rect.left ||
            clientX > rect.right ||
            clientY < rect.top ||
            clientY > rect.bottom
        ) {
            continue;
        }

        const parsedZ = Number.parseInt(
            getComputedStyle(element).zIndex,
            10
        );

        const zIndex = Number.isFinite(parsedZ)
            ? parsedZ
            : 0;

        if (element.dataset.geometryObject === "zone") {
            if (zIndex >= bestZoneZ) {
                bestZone = element;
                bestZoneZ = zIndex;
            }

        } else if (zIndex >= bestOtherZ) {
            bestOther = element;
            bestOtherZ = zIndex;
        }
    }

    return bestZone || bestOther;
}


// ============================================================================
// MODULE CLEANUP
// ============================================================================

export function destroyGeometry() {
    cancelGeometryInteraction({
        restore: true,
        clearTool: true
    });

    if (renderFrame !== null) {
        cancelAnimationFrame(renderFrame);
        renderFrame = null;
    }

    removePreview();
    clearRenderedGeometry();

    callbacks.onGeometryChange = null;
    overlayElement = null;
}
