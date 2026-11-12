
import { APP_CONFIG, state } from "./state.js";
import {
    getChartContainer,
    getOverlayElement,
    isInsideMainPlot
} from "./chart.js";
import {
    beginGeometryDraw,
    updateGeometryDraw,
    finishGeometryDraw,
    beginGeometryObjectInteraction,
    updateGeometryObjectInteraction,
    finishGeometryObjectInteraction,
    cancelGeometryInteraction,
    findEditableGeometryAtPoint,
    getGeometryTool,
    getActiveGeometryDrag,
    isGeometryInteractionActive
} from "./geometry.js";
import {
    pickPriceFromChart,
    pickAnchorFromChart
} from "./learning.js";
import { closeTradeDetail } from "./results.js";


// ============================================================================
// MODULE STATE
// ============================================================================

let chartContainer = null;
let chartPanel = null;
let overlayElement = null;
let listenerController = null;

let callbacks = {
    onRenderOverlay: null,
    onOutsideTouch: null
};

let activePointerId = null;
let activePointerType = null;
let passiveChartPointerId = null;
let wheelResetTimer = null;


// ============================================================================
// INTERNAL HELPERS
// ============================================================================

function isFiniteNumber(value) {
    return Number.isFinite(Number(value));
}

function isMobileTouchDevice() {
    return window.matchMedia(
        "(max-width: 900px) and (pointer: coarse)"
    ).matches;
}

function isTouchPointer(event) {
    return event?.pointerType === "touch" || (
        isMobileTouchDevice() && event?.pointerType !== "mouse"
    );
}

function getEventTarget(event) {
    return event?.target instanceof Element
        ? event.target
        : null;
}

function isIgnoredTarget(target) {
    if (!target) {
        return false;
    }

    return Boolean(target.closest(
        "button, input, select, textarea, a, " +
        "[data-gesture-ignore], " +
        "#tradeDetailPanel, #mt5-offline-popup, #loading-screen"
    ));
}

function createGeometryEvent(event, currentTarget = null) {
    return {
        button: 0,
        pointerId: event?.pointerId,
        pointerType: event?.pointerType,
        clientX: Number(event?.clientX),
        clientY: Number(event?.clientY),
        target: currentTarget || getEventTarget(event),
        currentTarget,
        preventDefault() {
            if (event?.cancelable) {
                event.preventDefault();
            }
        },
        stopPropagation() {
            event?.stopPropagation?.();
        }
    };
}

function clearLongPressTimer() {
    if (state.mobile.longPressTimer !== null) {
        clearTimeout(state.mobile.longPressTimer);
        state.mobile.longPressTimer = null;
    }
}

function clearClickSuppression() {
    if (state.mobile.clickSuppressionTimer !== null) {
        clearTimeout(state.mobile.clickSuppressionTimer);
        state.mobile.clickSuppressionTimer = null;
    }

    state.mobile.clickSuppression = null;
}

function rememberClickSuppression(point) {
    if (!point || !isFiniteNumber(point.clientX) || !isFiniteNumber(point.clientY)) {
        return;
    }

    clearClickSuppression();

    state.mobile.clickSuppression = {
        x: Number(point.clientX),
        y: Number(point.clientY),
        expiresAt: Date.now() + 450
    };

    state.mobile.clickSuppressionTimer = setTimeout(() => {
        state.mobile.clickSuppression = null;
        state.mobile.clickSuppressionTimer = null;
    }, 450);
}

function shouldSuppressClick(event) {
    const suppression = state.mobile.clickSuppression;

    if (!suppression) {
        return false;
    }

    if (Date.now() > suppression.expiresAt) {
        clearClickSuppression();
        return false;
    }

    const distance = Math.hypot(
        Number(event.clientX) - suppression.x,
        Number(event.clientY) - suppression.y
    );

    if (distance > 35) {
        return false;
    }

    clearClickSuppression();
    return true;
}

function findOverlayElementAtPoint(selector, clientX, clientY) {
    if (!overlayElement) {
        return null;
    }

    let bestElement = null;
    let bestZIndex = -Infinity;

    for (const element of overlayElement.querySelectorAll(selector)) {
        const rect = element.getBoundingClientRect();

        if (
            clientX < rect.left ||
            clientX > rect.right ||
            clientY < rect.top ||
            clientY > rect.bottom
        ) {
            continue;
        }

        const parsedZIndex = Number.parseInt(
            getComputedStyle(element).zIndex,
            10
        );

        const zIndex = Number.isFinite(parsedZIndex)
            ? parsedZIndex
            : 0;

        if (zIndex >= bestZIndex) {
            bestElement = element;
            bestZIndex = zIndex;
        }
    }

    return bestElement;
}

function requestOverlayRender() {
    if (typeof callbacks.onRenderOverlay !== "function") {
        return;
    }

    requestAnimationFrame(() => {
        callbacks.onRenderOverlay();
    });
}

function setPointerCapture(event) {
    if (
        chartPanel &&
        typeof chartPanel.setPointerCapture === "function" &&
        event?.pointerId !== undefined
    ) {
        try {
            chartPanel.setPointerCapture(event.pointerId);
        } catch (error) {
            console.warn("Pointer capture could not be set:", error);
        }
    }
}

function releasePointerCapture(pointerId) {
    if (
        pointerId === null ||
        !chartPanel ||
        typeof chartPanel.hasPointerCapture !== "function" ||
        typeof chartPanel.releasePointerCapture !== "function"
    ) {
        return;
    }

    try {
        if (chartPanel.hasPointerCapture(pointerId)) {
            chartPanel.releasePointerCapture(pointerId);
        }
    } catch (error) {
        console.warn("Pointer capture could not be released:", error);
    }
}

function resetPointerState() {
    const pointerId = activePointerId;

    activePointerId = null;
    activePointerType = null;
    passiveChartPointerId = null;

    releasePointerCapture(pointerId);
    document.body.style.userSelect = "";
    document.body.style.cursor = "";
}

function clearMobileGesture() {
    clearLongPressTimer();
    state.mobile.gesture = null;
}

function cancelActiveGeometry({ restore = true } = {}) {
    if (!isGeometryInteractionActive()) {
        return false;
    }

    cancelGeometryInteraction({
        restore,
        clearTool: false
    });

    return true;
}


// ============================================================================
// GEOMETRY INTERACTION CONTROL
// ============================================================================

function startGeometryDraw(event) {
    const started = beginGeometryDraw(event);

    if (!started) {
        return false;
    }

    activePointerId = event.pointerId ?? null;
    activePointerType = event.pointerType || "mouse";

    return true;
}

function startGeometryEdit(event, editableTarget) {
    if (!editableTarget) {
        return false;
    }

    const startEvent = createGeometryEvent(event, editableTarget);
    const started = beginGeometryObjectInteraction(startEvent);

    if (!started) {
        return false;
    }

    activePointerId = event.pointerId ?? null;
    activePointerType = event.pointerType || "mouse";

    return true;
}

function updateActiveGeometry(event) {
    const drag = getActiveGeometryDrag();

    if (!drag) {
        return false;
    }

    if (drag.mode === "draw") {
        return updateGeometryDraw(event);
    }

    return updateGeometryObjectInteraction(event);
}

function finishActiveGeometry(event) {
    const drag = getActiveGeometryDrag();

    if (!drag) {
        resetPointerState();
        return false;
    }

    try {
        if (drag.mode === "draw") {
            return finishGeometryDraw(event);
        }

        return finishGeometryObjectInteraction(event);
    } finally {
        resetPointerState();
    }
}

function cancelMobileAction() {
    clearLongPressTimer();

    if (state.mobile.gesture?.startedAction) {
        cancelActiveGeometry({ restore: true });
    }

    state.mobile.gesture = null;
    resetPointerState();
}

function startMobileAction(event, gesture) {
    if (!gesture || gesture.startedAction || isGeometryInteractionActive()) {
        return false;
    }

    const startEvent = createGeometryEvent({
        pointerId: event.pointerId,
        pointerType: event.pointerType,
        clientX: gesture.startX,
        clientY: gesture.startY,
        target: gesture.editableTarget || getEventTarget(event)
    }, gesture.editableTarget);

    let started = false;

    if (gesture.mode === "draw") {
        started = startGeometryDraw(startEvent);
    } else if (gesture.mode === "edit") {
        started = startGeometryEdit(startEvent, gesture.editableTarget);
    }

    if (!started) {
        gesture.moved = true;
        clearLongPressTimer();
        return false;
    }

    gesture.startedAction = true;
    activePointerId = event.pointerId;
    activePointerType = "touch";
    setPointerCapture(event);

    return true;
}


// ============================================================================
// MOBILE GESTURE CONTROL
// ============================================================================

function updateMobileGesture(event) {
    const gesture = state.mobile.gesture;

    if (!gesture || gesture.pointerId !== event.pointerId) {
        return false;
    }

    const distance = Math.hypot(
        event.clientX - gesture.startX,
        event.clientY - gesture.startY
    );

    if (gesture.mode === "anchor-hold") {
        if (!gesture.longPressReady) {
            if (distance > APP_CONFIG.mobile.moveThreshold) {
                gesture.moved = true;
                clearLongPressTimer();
            }

            return false;
        }

        if (event.cancelable) {
            event.preventDefault();
        }

        event.stopPropagation();
        return true;
    }

    if (!gesture.startedAction && !gesture.longPressReady) {
        if (distance > APP_CONFIG.mobile.moveThreshold) {
            gesture.moved = true;
            clearLongPressTimer();
        }

        return false;
    }

    if (!gesture.startedAction) {
        if (!startMobileAction(event, gesture)) {
            return false;
        }
    }

    const updated = updateActiveGeometry(event);

    if (event.cancelable) {
        event.preventDefault();
    }

    event.stopPropagation();
    return updated;
}

function finishMobileGesture(event) {
    const gesture = state.mobile.gesture;

    if (!gesture || gesture.pointerId !== event.pointerId) {
        return false;
    }

    clearLongPressTimer();

    if (gesture.mode === "anchor-hold") {
        if (gesture.longPressReady && !gesture.moved) {
            pickAnchorFromChart({
                clientX: event.clientX,
                clientY: event.clientY
            });

            rememberClickSuppression(event);
        }

        if (event.cancelable) {
            event.preventDefault();
        }

        state.mobile.gesture = null;
        resetPointerState();
        return true;
    }

    if (gesture.startedAction && isGeometryInteractionActive()) {
        finishActiveGeometry(event);

        if (event.cancelable) {
            event.preventDefault();
        }

        rememberClickSuppression(event);
        state.mobile.gesture = null;
        return true;
    }

    if (
        gesture.longPressReady ||
        gesture.moved
    ) {
        if (event.cancelable) {
            event.preventDefault();
        }

        rememberClickSuppression(event);
    }

    state.mobile.gesture = null;
    resetPointerState();
    return true;
}

function handleMobileLongPress(gesture) {
    if (state.mobile.gesture !== gesture || gesture.moved) {
        return;
    }

    gesture.longPressReady = true;

    if (
        gesture.mode === "edit" &&
        typeof navigator !== "undefined" &&
        typeof navigator.vibrate === "function"
    ) {
        navigator.vibrate(35);
    }
}

function startMobileGesture(event) {
    const target = getEventTarget(event);

    if (
        !isInsideMainPlot(event.clientX, event.clientY) ||
        isIgnoredTarget(target)
    ) {
        return false;
    }

    const anchorPickerActive = state.interaction.anchorPickerActive;
    const pricePickerActive = Boolean(state.interaction.activePriceTarget);
    const drawingTool = anchorPickerActive || pricePickerActive
        ? null
        : getGeometryTool();

    const editableTarget = (
        anchorPickerActive ||
        pricePickerActive ||
        drawingTool
    )
        ? null
        : findEditableGeometryAtPoint(event.clientX, event.clientY);

    let mode = "pan";

    if (anchorPickerActive) {
        mode = "anchor-hold";
    } else if (pricePickerActive) {
        mode = "picker";
    } else if (drawingTool) {
        mode = "draw";
    } else if (editableTarget) {
        mode = "edit";
    }

    const gesture = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
        mode,
        tool: drawingTool,
        editableTarget,
        moved: false,
        longPressReady: false,
        startedAction: false
    };

    state.mobile.gesture = gesture;

    if (["anchor-hold", "draw", "edit"].includes(mode)) {
        clearLongPressTimer();
        state.mobile.longPressTimer = setTimeout(() => {
            handleMobileLongPress(gesture);
        }, APP_CONFIG.mobile.longPressMs);
    }

    return true;
}


// ============================================================================
// MOBILE SIDEBAR CONTROL
// ============================================================================

function handleOutsideTouch(event) {
    if (!isMobileTouchDevice() || event.pointerType !== "touch") {
        return;
    }

    if (typeof callbacks.onOutsideTouch === "function") {
        callbacks.onOutsideTouch(event);
        return;
    }

    const sidebar = document.getElementById("sidebar");
    const sidebarToggle = document.getElementById("sidebarToggle");
    const target = getEventTarget(event);

    if (!sidebar || !sidebarToggle || !target) {
        return;
    }

    const icon = sidebarToggle.querySelector("img");

    if (sidebar.contains(target)) {
        if (
            sidebar.classList.contains("collapsed") &&
            !target.closest("#sidebarToggle")
        ) {
            sidebar.classList.remove("collapsed");

            if (icon) {
                icon.src = state.learning.mode
                    ? "images/left.png"
                    : "images/right.png";
            }
        }

        return;
    }

    sidebar.classList.add("collapsed");

    if (icon) {
        icon.src = "images/left.png";
    }
}


// ============================================================================
// POINTER EVENT HANDLERS
// ============================================================================

function handleDocumentPointerDown(event) {
    handleOutsideTouch(event);

    if (
        state.mobile.gesture &&
        state.mobile.gesture.pointerId !== event.pointerId
    ) {
        cancelMobileAction();
        clearClickSuppression();
        return;
    }

    if (
        activePointerId !== null &&
        activePointerId !== event.pointerId
    ) {
        cancelActiveGesture();
    }
}

function handleChartPointerDown(event) {
    if (
        !event.isPrimary ||
        (event.pointerType === "mouse" && event.button !== 0) ||
        isGeometryInteractionActive() ||
        activePointerId !== null
    ) {
        return;
    }

    if (isTouchPointer(event)) {
        startMobileGesture(event);
        return;
    }

    if (
        !isInsideMainPlot(event.clientX, event.clientY) ||
        isIgnoredTarget(getEventTarget(event))
    ) {
        return;
    }

    state.interaction.chartMouseDownX = event.clientX;
    state.interaction.chartMouseDownY = event.clientY;
    state.interaction.chartWasDragged = false;
    passiveChartPointerId = event.pointerId;

    if (
        state.interaction.anchorPickerActive ||
        state.interaction.activePriceTarget
    ) {
        return;
    }

    if (getGeometryTool()) {
        const started = startGeometryDraw(event);

        if (started) {
            setPointerCapture(event);
            event.preventDefault();
            event.stopPropagation();
        }

        return;
    }

    const editableTarget = findEditableGeometryAtPoint(
        event.clientX,
        event.clientY
    );

    if (editableTarget) {
        const started = startGeometryEdit(event, editableTarget);

        if (started) {
            setPointerCapture(event);
            event.preventDefault();
            event.stopPropagation();
        }
    }
}

function handleDocumentPointerMove(event) {
    if (
        state.mobile.gesture &&
        state.mobile.gesture.pointerId === event.pointerId
    ) {
        updateMobileGesture(event);
        return;
    }

    if (
        activePointerId !== null &&
        activePointerId === event.pointerId &&
        isGeometryInteractionActive()
    ) {
        const updated = updateActiveGeometry(event);

        if (updated) {
            if (event.cancelable) {
                event.preventDefault();
            }

            event.stopPropagation();
        }

        return;
    }

    if (
        passiveChartPointerId === event.pointerId &&
        event.pointerType === "mouse" &&
        (event.buttons & 1) === 1
    ) {
        const dx = event.clientX - state.interaction.chartMouseDownX;
        const dy = event.clientY - state.interaction.chartMouseDownY;

        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
            state.interaction.chartWasDragged = true;
        }
    }
}

function handleDocumentPointerUp(event) {
    if (
        state.mobile.gesture &&
        state.mobile.gesture.pointerId === event.pointerId
    ) {
        finishMobileGesture(event);
        return;
    }

    if (
        activePointerId !== null &&
        activePointerId === event.pointerId &&
        isGeometryInteractionActive()
    ) {
        finishActiveGeometry(event);

        if (event.cancelable) {
            event.preventDefault();
        }

        event.stopPropagation();
        return;
    }

    if (passiveChartPointerId === event.pointerId) {
        passiveChartPointerId = null;
    }
}

function handleDocumentPointerCancel(event) {
    if (
        state.mobile.gesture &&
        state.mobile.gesture.pointerId === event.pointerId
    ) {
        cancelMobileAction();
        return;
    }

    if (activePointerId === event.pointerId) {
        cancelActiveGesture();
    }

    if (passiveChartPointerId === event.pointerId) {
        passiveChartPointerId = null;
    }
}

function handleLostPointerCapture(event) {
    if (
        activePointerId !== null &&
        activePointerId === event.pointerId &&
        isGeometryInteractionActive()
    ) {
        cancelActiveGesture();
    }
}


// ============================================================================
// CHART CLICK AND WHEEL HANDLERS
// ============================================================================

function handleChartClick(event) {
    if (shouldSuppressClick(event)) {
        return;
    }

    if (state.interaction.chartWasDragged) {
        state.interaction.chartWasDragged = false;
        return;
    }

    if (!isInsideMainPlot(event.clientX, event.clientY)) {
        return;
    }

    if (state.interaction.anchorPickerActive) {
        pickAnchorFromChart(event);
        return;
    }

    if (state.interaction.activePriceTarget) {
        pickPriceFromChart(event);
        return;
    }

    const target = getEventTarget(event);

    if (target?.closest(".trade-box")) {
        return;
    }

    if (isMobileTouchDevice()) {
        const touchedTrade = findOverlayElementAtPoint(
            ".trade-box",
            event.clientX,
            event.clientY
        );

        if (touchedTrade) {
            touchedTrade.click();
            return;
        }
    }

    if (state.results.selectedTradeId === null) {
        return;
    }

    closeTradeDetail();
    requestOverlayRender();
}

function handleChartWheel() {
    state.interaction.chartWasDragged = true;

    if (wheelResetTimer !== null) {
        clearTimeout(wheelResetTimer);
    }

    wheelResetTimer = setTimeout(() => {
        state.interaction.chartWasDragged = false;
        wheelResetTimer = null;
    }, 100);
}


// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

export function initializeGestures({
    container = null,
    panel = null,
    overlay = null,
    onRenderOverlay = null,
    onOutsideTouch = null
} = {}) {
    destroyGestures();

    chartContainer = container || getChartContainer();
    chartPanel = panel || document.getElementById("chart-panel") || chartContainer;
    overlayElement = overlay || getOverlayElement();

    callbacks = {
        onRenderOverlay: typeof onRenderOverlay === "function"
            ? onRenderOverlay
            : null,
        onOutsideTouch: typeof onOutsideTouch === "function"
            ? onOutsideTouch
            : null
    };

    if (!chartContainer || !chartPanel) {
        console.error("Gestures could not initialize: chart container not found.");
        return false;
    }

    listenerController = new AbortController();
    const { signal } = listenerController;

    document.addEventListener(
        "pointerdown",
        handleDocumentPointerDown,
        { capture: true, signal }
    );

    document.addEventListener(
        "pointermove",
        handleDocumentPointerMove,
        { capture: true, passive: false, signal }
    );

    document.addEventListener(
        "pointerup",
        handleDocumentPointerUp,
        { capture: true, passive: false, signal }
    );

    document.addEventListener(
        "pointercancel",
        handleDocumentPointerCancel,
        { capture: true, signal }
    );

    chartPanel.addEventListener(
        "pointerdown",
        handleChartPointerDown,
        { capture: true, passive: false, signal }
    );

    chartPanel.addEventListener(
        "lostpointercapture",
        handleLostPointerCapture,
        { signal }
    );

    chartPanel.addEventListener(
        "click",
        handleChartClick,
        { signal }
    );

    chartContainer.addEventListener(
        "wheel",
        handleChartWheel,
        { passive: true, signal }
    );

    return true;
}


// ============================================================================
// PUBLIC INTERACTION API
// ============================================================================

export function cancelActiveGesture() {
    clearLongPressTimer();

    if (state.mobile.gesture?.startedAction || isGeometryInteractionActive()) {
        cancelActiveGeometry({ restore: true });
    }

    state.mobile.gesture = null;
    resetPointerState();
    return true;
}

export function isGestureActive() {
    return Boolean(
        state.mobile.gesture ||
        activePointerId !== null ||
        isGeometryInteractionActive()
    );
}

export function getActiveGesture() {
    if (state.mobile.gesture) {
        return state.mobile.gesture;
    }

    const drag = getActiveGeometryDrag();

    if (drag) {
        return {
            mode: drag.mode,
            pointerId: activePointerId,
            pointerType: activePointerType,
            geometry: drag
        };
    }

    return null;
}

export function destroyGestures() {
    if (listenerController) {
        listenerController.abort();
        listenerController = null;
    }

    clearLongPressTimer();
    clearClickSuppression();

    if (wheelResetTimer !== null) {
        clearTimeout(wheelResetTimer);
        wheelResetTimer = null;
    }

    if (isGeometryInteractionActive()) {
        cancelGeometryInteraction({
            restore: true,
            clearTool: false
        });
    }

    state.mobile.gesture = null;
    resetPointerState();

    chartContainer = null;
    chartPanel = null;
    overlayElement = null;

    callbacks = {
        onRenderOverlay: null,
        onOutsideTouch: null
    };

    return true;
}
