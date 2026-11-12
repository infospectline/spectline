
// ============================================================================
// APP CONFIGURATION
// ============================================================================

export const APP_CONFIG = Object.freeze({
    webAllowedTimeframes: Object.freeze([
        "M15",
        "M30",
        "H1",
        "H4",
        "D1",
        "W1"
    ]),

    mobile: Object.freeze({
        longPressMs: 380,
        moveThreshold: 9,
        editableHitRadius: 18
    })
});


// ============================================================================
// SHARED APPLICATION STATE
// ============================================================================

export const state = {
    app: {
        webMode: false,
        launchTicket: null
    },

    ui: {
        mt5OfflinePopupTimer: null
    },

    analysis: {
        manualAnalysisActive: false
    },

    connection: {
        socket: null
    },

    chart: {
        instance: null,
        candleSeries: null,
        lastData: null,
        viewportTimer: null,
        dataLoaded: false,
        activeSessionId: null,
        pendingSessionId: null
    },

    results: {
        selectedTradeId: null,
        selectedTradeDetail: null
    },

    learning: {
        mode: false,
        trades: [],
        selectedTradeId: null,
        nextTradeId: 1,
        selectedTradeIsNew: false,
        pendingAnchorCenter: false,
        nextZoneId: 1
    },

    interaction: {
        activeTool: null,
        activeZoneType: null,
        activeDrag: null,
        activePriceTarget: null,
        anchorPickerActive: false,
        chartMouseDownX: 0,
        chartMouseDownY: 0,
        chartWasDragged: false
    },

    mobile: {
        gesture: null,
        longPressTimer: null,
        clickSuppression: null,
        clickSuppressionTimer: null
    }
};
