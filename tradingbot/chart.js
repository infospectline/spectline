
import { state } from "./state.js";


// ============================================================================
// MODULE STATE
// ============================================================================

const VIEWPORT_UPDATE_DELAY = 1000;

const OVERLAY_ZOOM_SELECTOR = [
    "#overlay .trade-box",
    "#overlay .learning-geometry-rect",
    "#overlay .learning-geometry-resize-handle",
    "#overlay .learning-trade-setup-edge",
    "#overlay .learning-setup-level",
    "#overlay .learning-zone-rect"
].join(", ");

let chartContainer = null;
let overlayElement = null;
let viewUpdateFrame = null;
let pendingViewRange = null;

let callbacks = {
    onViewChange: null,
    onViewportChange: null
};

let resizeListenerRegistered = false;


// ============================================================================
// INTERNAL HELPERS
// ============================================================================

function getChartInstance() {
    return state.chart.instance;
}

function getCandlestickSeries() {
    return state.chart.candleSeries;
}

function isFiniteNumber(value) {
    return (
        value !== null &&
        value !== undefined &&
        value !== "" &&
        Number.isFinite(Number(value))
    );
}

function queueViewUpdate(range = null) {
    pendingViewRange = range;

    if (viewUpdateFrame !== null) {
        return;
    }

    viewUpdateFrame = requestAnimationFrame(() => {
        viewUpdateFrame = null;

        const latestRange = pendingViewRange;
        pendingViewRange = null;

        if (typeof callbacks.onViewChange === "function") {
            callbacks.onViewChange(latestRange);
        }
    });
}

function scheduleViewportUpdate(range) {
    if (
        !range ||
        !isFiniteNumber(range.from) ||
        !isFiniteNumber(range.to)
    ) {
        return;
    }

    clearTimeout(state.chart.viewportTimer);

    state.chart.viewportTimer = setTimeout(() => {
        state.chart.viewportTimer = null;

        if (typeof callbacks.onViewportChange === "function") {
            callbacks.onViewportChange({
                from: Math.floor(Number(range.from)),
                to: Math.floor(Number(range.to))
            });
        }
    }, VIEWPORT_UPDATE_DELAY);
}

function handleWindowResize() {
    resizeChart();
}

function handleOverlayWheelZoom(event) {
    if (!(event.target instanceof Element)) {
        return;
    }

    const interactiveObject = event.target.closest(
        OVERLAY_ZOOM_SELECTOR
    );

    if (!interactiveObject) {
        return;
    }

    const chart = getChartInstance();

    if (!chart || event.deltaY === 0) {
        return;
    }

    const timeScale = chart.timeScale();
    const visibleRange = timeScale.getVisibleLogicalRange();

    if (!visibleRange) {
        return;
    }

    const currentSpan =
        Number(visibleRange.to) - Number(visibleRange.from);

    if (!Number.isFinite(currentSpan) || currentSpan <= 0) {
        return;
    }

    const containerRect = chartContainer.getBoundingClientRect();

    const localX =
        event.clientX -
        containerRect.left -
        getChartPaneLeftOffset();

    const x = Math.max(
        0,
        Math.min(timeScale.width(), localX)
    );

    const logicalAtCursor = timeScale.coordinateToLogical(x);

    if (!isFiniteNumber(logicalAtCursor)) {
        return;
    }

    const zoomFactor = Math.exp(
        Math.max(
            -0.5,
            Math.min(0.5, event.deltaY * 0.0015)
        )
    );

    const totalBars = Math.max(
        10,
        Number(state.chart.lastData?.h1?.length ?? 50000)
    );

    const newSpan = Math.max(
        5,
        Math.min(totalBars, currentSpan * zoomFactor)
    );

    const cursorRatio = Math.max(
        0,
        Math.min(
            1,
            (logicalAtCursor - visibleRange.from) / currentSpan
        )
    );

    const newFrom = logicalAtCursor - cursorRatio * newSpan;
    const newTo = newFrom + newSpan;

    event.preventDefault();

    timeScale.setVisibleLogicalRange({
        from: newFrom,
        to: newTo
    });
}

function attachOverlayWheelZoom(element) {
    if (overlayElement === element) {
        return;
    }

    if (overlayElement) {
        overlayElement.removeEventListener(
            "wheel",
            handleOverlayWheelZoom
        );
    }

    overlayElement = element || null;

    if (overlayElement) {
        overlayElement.addEventListener(
            "wheel",
            handleOverlayWheelZoom,
            { passive: false }
        );
    }
}


// ============================================================================
// CHART INITIALIZATION
// ============================================================================

export function initializeChart({
    onViewChange = null,
    onViewportChange = null,
    overlay = null
} = {}) {
    callbacks = {
        onViewChange,
        onViewportChange
    };

    chartContainer = document.getElementById("chart");

    if (!chartContainer) {
        throw new Error('Chart container "#chart" was not found.');
    }

    attachOverlayWheelZoom(
        overlay || document.getElementById("overlay")
    );

    if (getChartInstance()) {
        resizeChart();
        return getChartInstance();
    }

    const library = window.LightweightCharts;

    if (!library) {
        throw new Error("LightweightCharts library is not available.");
    }

    const chart = library.createChart(chartContainer, {
        layout: {
            background: { color: "#111" },
            textColor: "#bbb"
        },

        localization: {
            timeFormatter: time => {
                const date = new Date(Number(time) * 1000);

                return date.toISOString()
                    .replace("T", " ")
                    .replace(".000Z", " UTC");
            }
        },

        rightPriceScale: {
            visible: false
        },

        leftPriceScale: {
            visible: true
        },

        grid: {
            vertLines: { color: "#222" },
            horzLines: { color: "#222" }
        },

        crosshair: {
            mode: library.CrosshairMode.Normal
        },

        timeScale: {
            timeVisible: true,
            secondsVisible: false,
            rightOffset: 5,

            tickMarkFormatter: time => {
                const date = new Date(Number(time) * 1000);

                const months = [
                    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
                ];

                return `${date.getUTCDate()} ${months[date.getUTCMonth()]}`;
            }
        },

        handleScroll: {
            mouseWheel: true,
            pressedMouseMove: true,
            horzTouchDrag: true,
            vertTouchDrag: true
        },

        handleScale: {
            mouseWheel: true,
            pinch: true,
            axisPressedMouseMove: true
        }
    });

    state.chart.instance = chart;
    state.chart.candleSeries = chart.addCandlestickSeries();

    chart.timeScale().subscribeVisibleTimeRangeChange(range => {
        queueViewUpdate(range);
        scheduleViewportUpdate(range);
    });

    if (!resizeListenerRegistered) {
        window.addEventListener("resize", handleWindowResize, {
            passive: true
        });

        resizeListenerRegistered = true;
    }

    resizeChart();

    return chart;
}


// ============================================================================
// CHART DATA
// ============================================================================

export function setChartCandles(candles, { fitContent = true } = {}) {
    const series = getCandlestickSeries();

    if (!series || !Array.isArray(candles)) {
        return false;
    }

    try {
        const chartData = candles.map(candle => {
            if (!candle || typeof candle !== "object") {
                throw new TypeError("A chart candle is invalid.");
            }

            return {
                time: candle.time,
                open: candle.open,
                high: candle.high,
                low: candle.low,
                close: candle.close
            };
        });

        series.setData(chartData);
        state.chart.dataLoaded = true;

        if (fitContent && getChartInstance()) {
            getChartInstance().timeScale().fitContent();
        }

        return true;

    } catch (error) {
        console.error("Failed to load chart candles:", error);
        return false;
    }
}

export function resetChartData() {
    clearTimeout(state.chart.viewportTimer);
    state.chart.viewportTimer = null;

    state.chart.dataLoaded = false;
    state.chart.lastData = null;

    const series = getCandlestickSeries();

    if (series) {
        series.setData([]);
    }
}

export function fitChartContent() {
    const chart = getChartInstance();

    if (!chart || !state.chart.dataLoaded) {
        return false;
    }

    chart.timeScale().fitContent();

    return true;
}


// ============================================================================
// CHART SIZE AND VIEWPORT
// ============================================================================

export function resizeChart() {
    const chart = getChartInstance();

    if (!chart || !chartContainer) {
        return false;
    }

    const width = Math.max(1, chartContainer.clientWidth);
    const height = Math.max(1, chartContainer.clientHeight);

    chart.applyOptions({
        width,
        height
    });

    queueViewUpdate();

    return true;
}


// ============================================================================
// CHART COORDINATES
// ============================================================================

export function timeCoord(time) {
    const chart = getChartInstance();

    if (!chart || !isFiniteNumber(time)) {
        return null;
    }

    const coordinate = chart.timeScale().timeToCoordinate(Number(time));

    return isFiniteNumber(coordinate)
        ? Number(coordinate)
        : null;
}

export function priceCoord(price) {
    const series = getCandlestickSeries();

    if (!series || !isFiniteNumber(price)) {
        return null;
    }

    const coordinate = series.priceToCoordinate(Number(price));

    return isFiniteNumber(coordinate)
        ? Number(coordinate)
        : null;
}

export function getChartPaneLeftOffset() {
    const chart = getChartInstance();

    if (!chart || !chartContainer) {
        return 0;
    }

    const chartWidth = chartContainer.clientWidth;
    const timeScaleWidth = chart.timeScale().width();

    return Math.max(0, chartWidth - timeScaleWidth);
}

export function getMainPlotRect() {
    const chart = getChartInstance();

    if (!chart || !chartContainer) {
        return null;
    }

    const containerRect = chartContainer.getBoundingClientRect();
    const plotWidth = Math.max(1, chart.timeScale().width());

    const tolerance = Math.max(6, plotWidth * 0.02);

    const paneCanvas = Array.from(
        chartContainer.querySelectorAll("canvas")
    )
        .map(canvas => canvas.getBoundingClientRect())
        .filter(rect =>
            rect.height > 80 &&
            Math.abs(rect.width - plotWidth) <= tolerance
        )
        .sort((a, b) => b.height - a.height)[0];

    if (paneCanvas) {
        return {
            left: paneCanvas.left,
            right: paneCanvas.right,
            top: paneCanvas.top,
            bottom: paneCanvas.bottom
        };
    }

    const left = containerRect.left + getChartPaneLeftOffset();

    return {
        left,
        right: left + plotWidth,
        top: containerRect.top,
        bottom: containerRect.bottom - 28
    };
}

export function isInsideMainPlot(clientX, clientY) {
    const rect = getMainPlotRect();

    if (
        !rect ||
        !isFiniteNumber(clientX) ||
        !isFiniteNumber(clientY)
    ) {
        return false;
    }

    return (
        clientX >= rect.left &&
        clientX < rect.right &&
        clientY >= rect.top &&
        clientY < rect.bottom
    );
}

export function geometryTimeToX(time) {
    const localX = timeCoord(time);

    if (localX === null) {
        return null;
    }

    return localX + getChartPaneLeftOffset();
}

export function geometryXToTime(x) {
    const chart = getChartInstance();

    if (!chart || !isFiniteNumber(x)) {
        return null;
    }

    const localX = Number(x) - getChartPaneLeftOffset();

    const rawTime = chart.timeScale().coordinateToTime(localX);

    if (rawTime == null) {
        return null;
    }

    const time = Number(rawTime);

    return Number.isFinite(time)
        ? Math.floor(time)
        : null;
}

export function geometryPriceToY(price) {
    return priceCoord(price);
}

export function geometryYToPrice(y) {
    const series = getCandlestickSeries();
    const container = chartContainer;
    const plotRect = getMainPlotRect();

    if (
        !series ||
        !container ||
        !plotRect ||
        !isFiniteNumber(y)
    ) {
        return null;
    }

    const containerRect = container.getBoundingClientRect();

    const plotY = Number(y) - (plotRect.top - containerRect.top);

    const price = series.coordinateToPrice(plotY);

    if (!isFiniteNumber(price)) {
        return null;
    }

    return Number(price);
}

export function snapXToCandleTime(x, candles = state.chart.lastData?.h1) {
    if (
        !Array.isArray(candles) ||
        !candles.length ||
        !isFiniteNumber(x)
    ) {
        return null;
    }

    let nearestTime = null;
    let nearestDistance = Infinity;

    for (const candle of candles) {
        if (!isFiniteNumber(candle?.time)) {
            continue;
        }

        const candleX = geometryTimeToX(Number(candle.time));

        if (candleX === null) {
            continue;
        }

        const distance = Math.abs(candleX - Number(x));

        if (distance < nearestDistance) {
            nearestDistance = distance;
            nearestTime = Number(candle.time);
        }
    }

    return nearestTime;
}


// ============================================================================
// MODULE ACCESSORS
// ============================================================================

export function getChartContainer() {
    return chartContainer;
}

export function getOverlayElement() {
    return overlayElement;
}
