
// ============================================================================
// IMPORTS
// ============================================================================

import { APP_CONFIG, state } from "./state.js";
import {
    initializeChart,
    setChartCandles,
    resetChartData,
    geometryTimeToX,
    geometryPriceToY
} from "./chart.js";
import {
    initializeResults,
    updateResultsContext,
    updateResultSummary,
    renderResultsOverlay,
    setSelectedTradeDetail,
    closeTradeDetail
} from "./results.js";
import {
    initializeLearning,
    setLearningTrades,
    updateLearningHeader,
    renderLearningAnchor,
    centerLearningChartOnAnchor,
    updateLearningGeometrySummary,
    openLearningMode,
    closeLearningMode,
    closeLearningTradeDetail,
    ensureLearningTrade,
    destroyLearning
} from "./learning.js";
import {
    initializeGeometry,
    setGeometryTool,
    renderLearningGeometry,
    destroyGeometry
} from "./geometry.js";
import {
    initializeGestures,
    cancelActiveGesture,
    destroyGestures
} from "./gestures.js";
import {
    initializeSocket,
    sendSocketAction,
    sendSocketMessage,
    isSocketConnected,
    destroySocket
} from "./socket.js";


// ============================================================================
// MODULE STATE
// ============================================================================

const elements = {};

let uiController = null;
let initialized = false;
let initializationPromise = null;
let overlayFrame = null;
let updateDataResetTimer = null;


// ============================================================================
// DOM INITIALIZATION
// ============================================================================

function cacheElements() {
    const ids = [
        "chart",
        "overlay",
        "chart-panel",
        "status",
        "loading-screen",
        "loading-bar",
        "loading-percent",
        "loading-message",
        "loading-title",
        "marketSelect",
        "timeframeSelect",
        "trainingButton",
        "testingButton",
        "learningButton",
        "updateDataButton",
        "manualAnalysisButton",
        "logoutButton",
        "showLevels",
        "showWinningTrades",
        "showLosingTrades",
        "showManualSetups",
        "resultsView",
        "learningView",
        "learningMainView",
        "learningListView",
        "learningDetailView",
        "sidebar",
        "sidebarToggle",
        "learningAddButton"
    ];

    for (const id of ids) {
        elements[id] = document.getElementById(id);
    }

    return Boolean(elements.chart && elements.overlay);
}

function bindEvent(element, eventName, handler, options = {}) {
    if (!element || !uiController) {
        return;
    }

    element.addEventListener(eventName, handler, {
        ...options,
        signal: uiController.signal
    });
}

function setStatus(message) {
    if (elements.status) {
        elements.status.textContent = String(message ?? "");
    }
}

function isMobileTouchDevice() {
    return window.matchMedia(
        "(max-width: 900px) and (pointer: coarse)"
    ).matches;
}


// ============================================================================
// WEB MODE
// ============================================================================

function applyWebTimeframeRestrictions() {
    if (!state.app.webMode || !elements.timeframeSelect) {
        return;
    }

    Array.from(elements.timeframeSelect.options).forEach(option => {
        if (!APP_CONFIG.webAllowedTimeframes.includes(option.value)) {
            option.remove();
        }
    });

    if (
        !APP_CONFIG.webAllowedTimeframes.includes(
            elements.timeframeSelect.value
        )
    ) {
        elements.timeframeSelect.value = "M15";
    }
}

async function loadWebMode() {
    try {
        const response = await fetch("/api/config", {
            headers: { Accept: "application/json" }
        });

        if (!response.ok) {
            throw new Error(`Configuration request failed: ${response.status}`);
        }

        const config = await response.json();
        state.app.webMode = Boolean(config?.web_mode);
    } catch (error) {
        state.app.webMode = false;
        console.error("Failed to load web mode:", error);
    }

    if (elements.logoutButton) {
        elements.logoutButton.style.display = state.app.webMode
            ? "block"
            : "none";
    }

    applyWebTimeframeRestrictions();
}

function readLaunchTicket() {
    const ticket = new URLSearchParams(
        window.location.hash.slice(1)
    ).get("ticket");

    state.app.launchTicket = ticket || null;

    if (ticket) {
        window.history.replaceState(
            null,
            "",
            window.location.pathname + window.location.search
        );
    }
}


// ============================================================================
// CHART AND OVERLAY RENDERING
// ============================================================================

function requestOverlayRender() {
    if (overlayFrame !== null) {
        return;
    }

    overlayFrame = requestAnimationFrame(() => {
        overlayFrame = null;
        renderOverlay();
    });
}

function formatPrice(price) {
    const value = Number(price);
    return Number.isFinite(value) ? value.toFixed(5) : "-";
}

function formatChartTime(timestamp) {
    if (timestamp === null || timestamp === undefined) {
        return "-";
    }

    const date = new Date(Number(timestamp) * 1000);

    if (Number.isNaN(date.getTime())) {
        return "-";
    }

    return date.toISOString()
        .replace("T", " ")
        .replace(".000Z", " UTC");
}

function clearManualSetupOverlays() {
    elements.overlay?.querySelectorAll(
        ".manual-setup-view-only"
    ).forEach(node => node.remove());
}

function appendManualSetupElement(className, styles = {}, text = null) {
    const element = document.createElement("div");
    element.className = className;

    for (const [property, value] of Object.entries(styles)) {
        element.style[property] = value;
    }

    if (text !== null) {
        element.textContent = String(text);
    }

    element.classList.add("manual-setup-view-only");
    elements.overlay.appendChild(element);
    return element;
}

function renderManualSetupOverlay(trade, index) {
    const geometry = trade?.trade_geometry;

    if (
        !geometry ||
        geometry.start_time == null ||
        geometry.end_time == null
    ) {
        return false;
    }

    const x1 = geometryTimeToX(geometry.start_time);
    const x2 = geometryTimeToX(geometry.end_time);

    if (x1 === null || x2 === null) {
        return false;
    }

    const prices = [
        geometry.breakout?.price,
        geometry.entry?.price,
        geometry.sl?.price,
        geometry.tp?.price
    ]
        .filter(value => value !== null && value !== undefined && Number.isFinite(Number(value)))
        .map(Number);

    if (!prices.length) {
        return false;
    }

    const yHigh = geometryPriceToY(Math.max(...prices));
    const yLow = geometryPriceToY(Math.min(...prices));

    if (yHigh === null || yLow === null) {
        return false;
    }

    const left = Math.min(x1, x2);
    const right = Math.max(x1, x2);
    const top = Math.min(yHigh, yLow);
    const bottom = Math.max(yHigh, yLow);
    const width = Math.max(12, right - left);
    const height = Math.max(20, bottom - top);

    appendManualSetupElement(
        "learning-geometry-rect learning-trade-setup-rect",
        {
            left: `${left}px`,
            top: `${top}px`,
            width: `${width}px`,
            height: `${height}px`
        }
    ).title = `MANUAL SETUP ${index + 1}`;

    const entryY = geometry.entry?.price != null
        ? geometryPriceToY(geometry.entry.price)
        : null;

    const areas = [
        [geometry.tp?.price, "learning-setup-profit-area"],
        [geometry.sl?.price, "learning-setup-risk-area"]
    ];

    for (const [targetPrice, className] of areas) {
        if (entryY === null || targetPrice == null) {
            continue;
        }

        const targetY = geometryPriceToY(targetPrice);

        if (targetY === null) {
            continue;
        }

        appendManualSetupElement(
            `learning-setup-area ${className}`,
            {
                left: `${left}px`,
                top: `${Math.min(entryY, targetY)}px`,
                width: `${width}px`,
                height: `${Math.max(1, Math.abs(targetY - entryY))}px`
            }
        );
    }

    const levels = [
        ["breakout", "BREAKOUT", "learning-setup-breakout"],
        ["entry", "ENTRY", "learning-setup-entry"],
        ["sl", "SL", "learning-setup-sl"],
        ["tp", "TP", "learning-setup-tp"]
    ];

    for (const [key, label, className] of levels) {
        const point = geometry[key];

        if (!point || point.price == null) {
            continue;
        }

        const y = geometryPriceToY(point.price);

        if (y === null) {
            continue;
        }

        appendManualSetupElement(
            `learning-setup-level ${className}`,
            {
                left: `${left}px`,
                width: `${width}px`,
                top: `${y - 2}px`
            }
        ).title = `${label} ${formatPrice(point.price)}`;

        appendManualSetupElement(
            "learning-setup-label",
            {
                left: `${left + 6}px`,
                top: `${y - 18}px`
            },
            `${label} ${formatPrice(point.price)} | ${formatChartTime(point.time)}`
        );
    }

    return true;
}

function renderManualSetupsOverlay() {
    elements.overlay.replaceChildren();

    state.learning.trades.forEach((trade, index) => {
        renderManualSetupOverlay(trade, index);
    });
}

function renderOverlay() {
    const data = state.chart.lastData;

    if (!elements.overlay || !data?.ready) {
        return false;
    }

    if (elements.showManualSetups?.classList.contains("active")) {
        renderManualSetupsOverlay();
        return true;
    }

    clearManualSetupOverlays();

    renderResultsOverlay(data, {
        showLevels: Boolean(elements.showLevels?.classList.contains("active")),
        showWinningTrades: Boolean(elements.showWinningTrades?.classList.contains("active")),
        showLosingTrades: Boolean(elements.showLosingTrades?.classList.contains("active"))
    });

    const detailVisible =
        state.learning.mode &&
        elements.learningDetailView?.style.display !== "none" &&
        state.learning.selectedTradeId !== null;

    if (detailVisible) {
        renderLearningAnchor();
        renderLearningGeometry();
    }

    return true;
}

function resetChartView() {
    resetChartData();
    closeTradeDetail();

    if (elements.overlay) {
        elements.overlay.replaceChildren();
    }
}

function syncSelectorsFromBackend(data) {
    const symbol = data?.symbol;
    const timeframe = data?.timeframe;

    if (symbol && elements.marketSelect) {
        const exists = Array.from(elements.marketSelect.options)
            .some(option => option.value === symbol);

        if (exists) {
            elements.marketSelect.value = symbol;
        }
    }

    if (timeframe && elements.timeframeSelect) {
        const exists = Array.from(elements.timeframeSelect.options)
            .some(option => option.value === timeframe);

        if (exists) {
            elements.timeframeSelect.value = timeframe;
        }
    }

    updateLearningHeader();
}


// ============================================================================
// BACKEND DATA RENDERING
// ============================================================================

function updateLoading(data) {
    const screen = elements["loading-screen"];

    if (!screen) {
        return;
    }

    const progress = Math.max(
        0,
        Math.min(100, Number(data?.startup_progress ?? 0))
    );

    const symbol = data?.symbol || elements.marketSelect?.value || "";
    const isLoading =
        data?.startup_stage === "Loading market data" ||
        data?.startup_stage === "Starting" ||
        data?.loading === true;

    if (elements["loading-title"]) {
        elements["loading-title"].textContent = symbol
            ? `Loading ${symbol} emulator`
            : "Loading emulator";
    }

    if (elements["loading-bar"]) {
        elements["loading-bar"].style.width = `${progress}%`;
    }

    if (elements["loading-percent"]) {
        elements["loading-percent"].textContent = `${progress}%`;
    }

    if (elements["loading-message"]) {
        elements["loading-message"].textContent = data?.startup_message || "Loading...";
    }

    screen.style.display = !isLoading && data?.ready && data?.h1?.length
        ? "none"
        : "flex";
}

function getStatusText(data) {
    const symbol = data?.symbol || elements.marketSelect?.value || "";
    const timeframe = data?.timeframe || elements.timeframeSelect?.value || "";
    const price = Number(data?.current_price ?? 0);
    const levels = Array.isArray(data?.levels) ? data.levels.length : 0;
    const trades = Array.isArray(data?.trades) ? data.trades.length : 0;
    const analysisStatus = data?.analysis_status || "idle";

    let status = "Waiting for analysis";

    if (data?.startup_stage === "Loading market data") {
        status = "Loading market data";
    } else if (analysisStatus === "training") {
        status = "TRAINING running";
    } else if (analysisStatus === "testing") {
        status = "TESTING running";
    } else if (data?.analysis_phase === "training_completed") {
        status = "TRAINING completed";
    } else if (data?.analysis_phase === "testing_completed") {
        status = "TESTING completed";
    } else if (data?.analysis_phase === "training_error") {
        status = "TRAINING error";
    } else if (data?.analysis_phase === "testing_error") {
        status = "TESTING error";
    } else if (analysisStatus === "manual_analysis") {
        status = "MANUAL ANALYSIS running";
    } else if (data?.analysis_phase === "manual_analysis_completed") {
        status = "MANUAL ANALYSIS completed";
    } else if (data?.analysis_phase === "manual_analysis_error") {
        status = "MANUAL ANALYSIS error";
    } else if (analysisStatus === "running") {
        status = "Analysis running";
    } else if (analysisStatus === "completed") {
        status = "Analysis completed";
    } else if (analysisStatus === "paused") {
        status = "Analysis paused";
    }

    const safePrice = Number.isFinite(price) ? price : 0;

    return `${symbol} | ${timeframe} | ${status} | Price ${safePrice.toFixed(5)} | Levels ${levels} | Trades ${trades}`;
}

function processBackendData(data) {
    if (!data || typeof data !== "object") {
        return false;
    }

    updateLoading(data);

    const isLoading =
        data?.startup_stage === "Loading market data" ||
        data?.startup_stage === "Starting" ||
        data?.loading === true;

    if (isLoading) {
        setStatus(data?.startup_message || "Loading market data...");
        return true;
    }

    syncSelectorsFromBackend(data);
    updateResultsContext(data);

    if (
        !data.ready ||
        !Array.isArray(data.h1) ||
        !Array.isArray(data.line)
    ) {
        return false;
    }

    const incomingSessionId = data.session_id ||
        `${data.symbol || ""}|${data.timeframe || ""}`;

    if (state.chart.activeSessionId !== incomingSessionId) {
        state.chart.activeSessionId = incomingSessionId;
        state.chart.pendingSessionId = null;
        resetChartView();
    }

    state.chart.lastData = data;

    if (!state.chart.dataLoaded) {
        const loaded = setChartCandles(data.h1, { fitContent: true });

        if (!loaded) {
            setStatus("Chart data could not be loaded.");
            return false;
        }

        if (
            state.learning.mode &&
            elements.learningDetailView?.style.display !== "none"
        ) {
            requestAnimationFrame(() => {
                centerLearningChartOnAnchor();
                renderLearningAnchor();
            });
        }
    }

    updateResultSummary(data);
    setStatus(getStatusText(data));
    requestOverlayRender();

    return true;
}

function updateManualAnalysisButton(data) {
    const button = elements.manualAnalysisButton;

    if (!button || !state.analysis.manualAnalysisActive) {
        return;
    }

    if (data?.analysis_status === "manual_analysis") {
        const progress = Math.max(
            0,
            Math.min(100, Number(data?.manual_analysis_progress ?? 0))
        );

        button.textContent = `ANALYSIS ${progress}%`;
        return;
    }

    if (
        data?.analysis_phase === "manual_analysis_completed" ||
        data?.analysis_phase === "manual_analysis_error" ||
        data?.analysis_phase === "manual_analysis_paused"
    ) {
        button.textContent = "ANALYSIS";
        state.analysis.manualAnalysisActive = false;
    }
}

function handleAnalysisFinished() {
    if (elements.trainingButton) {
        elements.trainingButton.disabled = false;
    }

    if (elements.testingButton) {
        elements.testingButton.disabled = false;
    }

    if (elements.manualAnalysisButton) {
        elements.manualAnalysisButton.disabled = false;
    }

    if (elements.learningAddButton) {
        elements.learningAddButton.disabled = false;
    }
}


// ============================================================================
// APPLICATION ACTIONS
// ============================================================================

function ensureBackendReady() {
    if (!isSocketConnected()) {
        setStatus("Backend is not connected.");
        return false;
    }

    if (!state.chart.lastData?.ready) {
        setStatus("Chart is not ready.");
        return false;
    }

    return true;
}

function handleTraining() {
    if (!ensureBackendReady()) {
        return;
    }

    elements.trainingButton.disabled = true;
    if (elements.testingButton) {
        elements.testingButton.disabled = true;
    }

    if (!sendSocketAction("training")) {
        handleAnalysisFinished();
        setStatus("Training request could not be sent.");
        return;
    }

    setStatus(`Training ${elements.marketSelect?.value || ""} ${elements.timeframeSelect?.value || ""}...`);
}

function handleTesting() {
    if (!ensureBackendReady()) {
        return;
    }

    elements.testingButton.disabled = true;
    if (elements.trainingButton) {
        elements.trainingButton.disabled = true;
    }

    if (!sendSocketAction("testing")) {
        handleAnalysisFinished();
        setStatus("Testing request could not be sent.");
        return;
    }

    setStatus(`Testing ${elements.marketSelect?.value || ""} ${elements.timeframeSelect?.value || ""}...`);
}

function handleMarketChange() {
    if (!isSocketConnected()) {
        setStatus("Backend is not connected.");
        return;
    }

    const symbol = elements.marketSelect?.value || "";
    const timeframe = elements.timeframeSelect?.value || "";

    if (!sendSocketAction("symbol", { symbol })) {
        setStatus("Market change request could not be sent.");
        return;
    }

    state.chart.pendingSessionId = `${symbol}|${timeframe}`;
    resetChartView();
    setStatus(`Loading ${symbol} ${timeframe}...`);
    updateLearningHeader();
}

function handleTimeframeChange() {
    const timeframe = elements.timeframeSelect?.value || "";

    if (
        state.app.webMode &&
        !APP_CONFIG.webAllowedTimeframes.includes(timeframe)
    ) {
        elements.timeframeSelect.value = "M15";
        return;
    }

    if (!isSocketConnected()) {
        setStatus("Backend is not connected.");
        return;
    }

    const symbol = elements.marketSelect?.value || "";

    if (!sendSocketAction("timeframe", { timeframe })) {
        setStatus("Timeframe change request could not be sent.");
        return;
    }

    state.chart.pendingSessionId = `${symbol}|${timeframe}`;

    if (
        state.learning.mode &&
        elements.learningDetailView?.style.display !== "none" &&
        state.learning.selectedTradeId !== null
    ) {
        state.learning.pendingAnchorCenter = true;
    }

    resetChartView();
    setStatus(`Loading ${symbol} ${timeframe}...`);
    updateLearningHeader();
}

function handleUpdateData() {
    if (!isSocketConnected()) {
        setStatus("Backend is not connected.");
        return;
    }

    if (!sendSocketAction("update_data")) {
        setStatus("Data update request could not be sent.");
        return;
    }

    if (elements.updateDataButton) {
        elements.updateDataButton.disabled = true;
    }

    setStatus("Updating market data...");

    if (updateDataResetTimer !== null) {
        clearTimeout(updateDataResetTimer);
    }

    updateDataResetTimer = setTimeout(() => {
        updateDataResetTimer = null;

        if (elements.updateDataButton) {
            elements.updateDataButton.disabled = false;
        }
    }, 1000);
}

function handleManualAnalysis() {
    if (!isSocketConnected()) {
        setStatus("Backend is not connected.");
        return;
    }

    state.analysis.manualAnalysisActive = true;

    if (elements.manualAnalysisButton) {
        elements.manualAnalysisButton.disabled = true;
        elements.manualAnalysisButton.textContent = "ANALYSIS 0%";
    }

    if (elements.learningAddButton) {
        elements.learningAddButton.disabled = true;
    }

    if (elements.trainingButton) {
        elements.trainingButton.disabled = true;
    }

    if (elements.testingButton) {
        elements.testingButton.disabled = true;
    }

    if (!sendSocketAction("manual_analysis")) {
        state.analysis.manualAnalysisActive = false;
        handleAnalysisFinished();
        if (elements.manualAnalysisButton) {
            elements.manualAnalysisButton.textContent = "ANALYSIS";
        }
        setStatus("Manual analysis request could not be sent.");
        return;
    }

    setStatus("Analyzing manually entered trades...");
}


// ============================================================================
// SIDEBAR AND NAVIGATION
// ============================================================================

function setSidebarIcon(source) {
    const image = elements.sidebarToggle?.querySelector("img");

    if (image) {
        image.src = source;
    }
}

function handleSidebarToggle() {
    const sidebar = elements.sidebar;

    if (!sidebar || !elements.sidebarToggle) {
        return;
    }

    if (
        isMobileTouchDevice() &&
        sidebar.classList.contains("collapsed")
    ) {
        sidebar.classList.remove("collapsed");
        setSidebarIcon(state.learning.mode ? "images/left.png" : "images/right.png");
        return;
    }

    if (state.learning.mode) {
        if (elements.learningDetailView?.style.display !== "none") {
            closeLearningTradeDetail();
            return;
        }

        if (elements.learningListView?.style.display !== "none") {
            openLearningMode();
            return;
        }

        closeLearningMode();
        return;
    }

    sidebar.classList.toggle("collapsed");
    setSidebarIcon(
        sidebar.classList.contains("collapsed")
            ? "images/left.png"
            : "images/right.png"
    );
}


// ============================================================================
// USER INTERFACE EVENTS
// ============================================================================

function initializeUIEvents() {
    if (uiController) {
        uiController.abort();
    }

    uiController = new AbortController();

    bindEvent(elements.trainingButton, "click", handleTraining);
    bindEvent(elements.testingButton, "click", handleTesting);
    bindEvent(elements.marketSelect, "change", handleMarketChange);
    bindEvent(elements.timeframeSelect, "change", handleTimeframeChange);
    bindEvent(elements.updateDataButton, "click", handleUpdateData);
    bindEvent(elements.manualAnalysisButton, "click", handleManualAnalysis);
    bindEvent(elements.sidebarToggle, "click", handleSidebarToggle);

    for (const id of [
        "showLevels",
        "showWinningTrades",
        "showLosingTrades",
        "showManualSetups"
    ]) {
        bindEvent(elements[id], "click", () => {
            elements[id].classList.toggle("active");
            requestOverlayRender();
        });
    }

    bindEvent(elements.logoutButton, "click", () => {
        const dashboardUrl = window.SPECTLINE_DASHBOARD_URL ||
            "https://www.spectline.com/dashboard";
        window.location.href = dashboardUrl;
    });
}


// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

function initializeChartModule() {
    initializeChart({
        overlay: elements.overlay,
        onViewChange: requestOverlayRender,
        onViewportChange: range => {
            sendSocketAction("viewport", range);
        }
    });
}

function initializeResultsModule() {
    initializeResults({
        overlay: elements.overlay,
        onTradeDetailRequest: tradeId => {
            sendSocketAction("trade_detail", {
                trade_id: tradeId
            });
        }
    });
}

function initializeGeometryModule() {
    initializeGeometry({
        overlay: elements.overlay,
        onGeometryChange: () => {
            renderLearningGeometry();
            updateLearningGeometrySummary();
            requestOverlayRender();
        }
    });
}

function initializeLearningModule() {
    initializeLearning({
        onSendAction: payload => sendSocketMessage(payload),
        onEnsureGeometry: trade => ensureLearningTrade(trade),
        onRenderGeometry: () => {
            renderLearningGeometry();
            requestOverlayRender();
        },
        onUpdateGeometrySummary: updateLearningGeometrySummary,
        onSetGeometryTool: (tool, zoneType) => {
            setGeometryTool(tool, zoneType);
        },
        onClearTradeSetup: () => {
            elements.overlay?.querySelectorAll(
                ".setup-line, .setup-point, .setup-label, .setup-level-highlight"
            ).forEach(node => node.remove());
        },
        onCancelInteraction: cancelActiveGesture
    });
}

function initializeGesturesModule() {
    initializeGestures({
        container: elements.chart,
        panel: elements["chart-panel"] || elements.chart,
        overlay: elements.overlay,
        onRenderOverlay: requestOverlayRender
    });
}

function initializeSocketModule() {
    initializeSocket({
        launchTicket: state.app.launchTicket,
        handlers: {
            onStatusChange: status => {
                if (status === "connected") {
                    setStatus("Connected");
                } else if (status === "connecting") {
                    setStatus("Connecting to backend...");
                } else if (status === "reconnecting") {
                    setStatus("Reconnecting to backend...");
                } else if (status === "disconnected") {
                    setStatus("Disconnected");
                } else if (status === "error") {
                    setStatus("Connection error");
                }
            },
            onData: processBackendData,
            onManualAnalysisUpdate: updateManualAnalysisButton,
            onAnalysisFinished: handleAnalysisFinished,
            onTradeDetail: data => {
                setSelectedTradeDetail(
                    Number(data.trade_id),
                    data.trade
                );
                requestOverlayRender();
            },
            onTradeDetailError: (data, error) => {
                console.error("Trade detail error:", error, data);
                setStatus("Trade details could not be loaded.");
            },
            onLearningTrades: trades => {
                setLearningTrades(trades);
                requestOverlayRender();
            },
            onLearningDataRefresh: data => {
                processBackendData(data);
            },
            onProtocolError: error => {
                console.error("WebSocket protocol error:", error);
            }
        }
    });
}

export async function initializeApp() {
    if (initialized) {
        return true;
    }

    if (initializationPromise) {
        return initializationPromise;
    }

    initializationPromise = (async () => {
        try {
            if (!cacheElements()) {
                console.error('Application initialization failed: required "#chart" or "#overlay" element is missing.');
                return false;
            }

            readLaunchTicket();

            initializeChartModule();
            initializeResultsModule();
            initializeGeometryModule();
            initializeLearningModule();
            initializeGesturesModule();
            initializeUIEvents();

            await loadWebMode();
            initializeSocketModule();

            initialized = true;
            return true;
        } catch (error) {
            console.error("Application initialization failed:", error);
            destroyApp();
            return false;
        } finally {
            initializationPromise = null;
        }
    })();

    return initializationPromise;
}


// ============================================================================
// MODULE CLEANUP
// ============================================================================

export function destroyApp() {
    if (uiController) {
        uiController.abort();
        uiController = null;
    }

    if (overlayFrame !== null) {
        cancelAnimationFrame(overlayFrame);
        overlayFrame = null;
    }

    if (updateDataResetTimer !== null) {
        clearTimeout(updateDataResetTimer);
        updateDataResetTimer = null;
    }

    cancelActiveGesture();
    destroyGestures();
    destroyLearning();
    destroyGeometry();
    destroySocket();

    initialized = false;
    return true;
}


// ============================================================================
// APPLICATION STARTUP
// ============================================================================

function startApplication() {
    initializeApp();
}

if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", startApplication, { once: true });
} else {
    startApplication();
}
