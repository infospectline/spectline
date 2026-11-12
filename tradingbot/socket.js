
// ============================================================================
// IMPORTS
// ============================================================================

import { state } from "./state.js";


// ============================================================================
// MODULE CONFIGURATION
// ============================================================================

const DEFAULT_RECONNECT_DELAY_MS = 1500;
const DEFAULT_SOCKET_PATH = "/ws";

const COMPLETED_ANALYSIS_PHASES = new Set([
    "training_completed",
    "testing_completed",
    "training_error",
    "testing_error",
    "manual_analysis_completed",
    "manual_analysis_error"
]);

const LEARNING_ACTIONS = new Set([
    "loaded",
    "saved",
    "deleted"
]);

const DEFAULT_CALLBACKS = Object.freeze({
    onStatusChange: null,
    onOpen: null,
    onClose: null,
    onError: null,
    onReconnectScheduled: null,
    onMessage: null,
    onData: null,
    onManualAnalysisUpdate: null,
    onAnalysisFinished: null,
    onTradeDetail: null,
    onTradeDetailError: null,
    onLearningTrades: null,
    onLearningDataRefresh: null,
    onProtocolError: null
});


// ============================================================================
// MODULE STATE
// ============================================================================

let socketOptions = {
    url: null,
    baseUrl: null,
    path: DEFAULT_SOCKET_PATH,
    launchTicket: null,
    reconnectDelayMs: DEFAULT_RECONNECT_DELAY_MS,
    autoConnect: true
};

let callbacks = { ...DEFAULT_CALLBACKS };

let reconnectTimer = null;
let reconnectAttempts = 0;
let connectionGeneration = 0;
let reconnectEnabled = false;
let connectionStatus = "idle";


// ============================================================================
// INTERNAL HELPERS
// ============================================================================

function isFunction(value) {
    return typeof value === "function";
}

function invokeCallback(name, ...args) {
    const callback = callbacks[name];

    if (!isFunction(callback)) {
        return undefined;
    }

    try {
        return callback(...args);
    } catch (error) {
        console.error(`Socket callback "${name}" failed:`, error);
        return undefined;
    }
}

function setConnectionStatus(status, details = null) {
    connectionStatus = status;
    invokeCallback("onStatusChange", status, details);
}

function clearReconnectTimer() {
    if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
}

function getLaunchTicket() {
    if (
        socketOptions.launchTicket !== null &&
        socketOptions.launchTicket !== undefined
    ) {
        return String(socketOptions.launchTicket);
    }

    if (
        state.app?.launchTicket !== null &&
        state.app?.launchTicket !== undefined
    ) {
        return String(state.app.launchTicket);
    }

    if (typeof window !== "undefined") {
        try {
            return new URLSearchParams(
                window.location.hash.slice(1)
            ).get("ticket") || "";
        } catch (error) {
            console.warn("Launch ticket could not be read:", error);
        }
    }

    return "";
}

function normalizeWebSocketProtocol(url) {
    if (url.protocol === "https:") {
        url.protocol = "wss:";
    } else if (url.protocol === "http:") {
        url.protocol = "ws:";
    }

    if (url.protocol !== "ws:" && url.protocol !== "wss:") {
        throw new Error(`Unsupported WebSocket protocol: ${url.protocol}`);
    }

    return url;
}

function buildSocketUrl() {
    if (typeof window === "undefined" || !window.location) {
        throw new Error("WebSocket connection requires a browser location.");
    }

    let socketUrl;

    if (socketOptions.url) {
        socketUrl = new URL(
            socketOptions.url,
            window.location.href
        );
    } else {
        let backendBase = socketOptions.baseUrl;

        if (!backendBase) {
            backendBase = window.location.protocol === "file:"
                ? "http://127.0.0.1:8000"
                : `${window.location.protocol}//${window.location.host}`;
        }

        socketUrl = new URL(backendBase, window.location.href);
        normalizeWebSocketProtocol(socketUrl);

        socketUrl.pathname = socketOptions.path.startsWith("/")
            ? socketOptions.path
            : `/${socketOptions.path}`;
    }

    normalizeWebSocketProtocol(socketUrl);
    socketUrl.searchParams.set("ticket", getLaunchTicket());

    return socketUrl.toString();
}

function isCurrentConnection(socket, generation) {
    return (
        state.connection.socket === socket &&
        connectionGeneration === generation
    );
}

function isAnalysisFinished(data) {
    return COMPLETED_ANALYSIS_PHASES.has(
        String(data?.analysis_phase || "")
    );
}

function isLearningTradesMessage(data) {
    return (
        LEARNING_ACTIONS.has(String(data?.learning_action || "")) &&
        Array.isArray(data?.learning_trades)
    );
}

function handleTradeDetailMessage(data) {
    if (!data.success) {
        const error = data.error || "The backend could not load trade details.";

        console.error("Trade detail error:", error);
        invokeCallback("onTradeDetailError", data, error);
        return;
    }

    invokeCallback("onTradeDetail", data);
}

function handleLearningTradesMessage(data) {
    invokeCallback(
        "onLearningTrades",
        data.learning_trades,
        data
    );

    if (
        data.ready === true &&
        Array.isArray(data.h1) &&
        Array.isArray(data.line)
    ) {
        state.chart.lastData = data;
        invokeCallback("onLearningDataRefresh", data);
    }
}

function handleIncomingMessage(event) {
    let data;

    try {
        data = typeof event.data === "string"
            ? JSON.parse(event.data)
            : event.data;
    } catch (error) {
        console.error("Invalid JSON received from backend:", error);
        invokeCallback("onProtocolError", error, event);
        return;
    }

    if (!data || typeof data !== "object" || Array.isArray(data)) {
        const error = new Error("Backend message must be a JSON object.");
        invokeCallback("onProtocolError", error, event);
        return;
    }

    invokeCallback("onMessage", data, event);
    invokeCallback("onManualAnalysisUpdate", data);

    if (isAnalysisFinished(data)) {
        invokeCallback("onAnalysisFinished", data);
    }

    if (data.type === "trade_detail") {
        handleTradeDetailMessage(data);
        return;
    }

    if (isLearningTradesMessage(data)) {
        handleLearningTradesMessage(data);
        return;
    }

    invokeCallback("onData", data);
}

function scheduleReconnect() {
    if (!reconnectEnabled || reconnectTimer !== null) {
        return;
    }

    const delay = Math.max(
        250,
        Number(socketOptions.reconnectDelayMs) || DEFAULT_RECONNECT_DELAY_MS
    );

    reconnectAttempts++;
    setConnectionStatus("reconnecting", {
        delay,
        attempt: reconnectAttempts
    });

    invokeCallback("onReconnectScheduled", {
        delay,
        attempt: reconnectAttempts
    });

    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;

        if (reconnectEnabled) {
            connectSocket();
        }
    }, delay);
}

function closeCurrentSocket({ code = 1000, reason = "Client disconnect" } = {}) {
    const currentSocket = state.connection.socket;

    if (!currentSocket) {
        return;
    }

    state.connection.socket = null;
    connectionGeneration++;

    try {
        if (
            currentSocket.readyState === WebSocket.CONNECTING ||
            currentSocket.readyState === WebSocket.OPEN
        ) {
            currentSocket.close(code, reason);
        }
    } catch (error) {
        console.warn("WebSocket close failed:", error);
    }
}


// ============================================================================
// CONNECTION LIFECYCLE
// ============================================================================

export function initializeSocket({
    url = null,
    baseUrl = null,
    path = DEFAULT_SOCKET_PATH,
    launchTicket = null,
    reconnectDelayMs = DEFAULT_RECONNECT_DELAY_MS,
    autoConnect = true,
    handlers = {}
} = {}) {
    destroySocket();

    socketOptions = {
        url,
        baseUrl,
        path: String(path || DEFAULT_SOCKET_PATH),
        launchTicket,
        reconnectDelayMs: Number(reconnectDelayMs) || DEFAULT_RECONNECT_DELAY_MS,
        autoConnect: Boolean(autoConnect)
    };

    const validHandlers = handlers && typeof handlers === "object"
        ? Object.fromEntries(
            Object.entries(handlers).filter(([, handler]) => isFunction(handler))
        )
        : {};

    callbacks = {
        ...DEFAULT_CALLBACKS,
        ...validHandlers
    };

    reconnectEnabled = socketOptions.autoConnect;
    reconnectAttempts = 0;
    setConnectionStatus("idle");

    if (!socketOptions.autoConnect) {
        return null;
    }

    return connectSocket();
}

export function connectSocket() {
    if (typeof WebSocket === "undefined") {
        const error = new Error("WebSocket is not supported in this environment.");
        setConnectionStatus("error", error);
        invokeCallback("onError", error);
        return null;
    }

    const existingSocket = state.connection.socket;

    if (
        existingSocket &&
        (
            existingSocket.readyState === WebSocket.CONNECTING ||
            existingSocket.readyState === WebSocket.OPEN
        )
    ) {
        return existingSocket;
    }

    clearReconnectTimer();
    reconnectEnabled = true;

    let socketUrl;

    try {
        socketUrl = buildSocketUrl();
    } catch (error) {
        console.error("WebSocket URL could not be created:", error);
        setConnectionStatus("error", error);
        invokeCallback("onError", error);
        scheduleReconnect();
        return null;
    }

    let socket;

    try {
        socket = new WebSocket(socketUrl);
    } catch (error) {
        console.error("WebSocket connection could not be created:", error);
        setConnectionStatus("error", error);
        invokeCallback("onError", error);
        scheduleReconnect();
        return null;
    }

    const generation = ++connectionGeneration;
    state.connection.socket = socket;

    setConnectionStatus("connecting", {
        url: socketUrl,
        attempt: reconnectAttempts + 1
    });

    socket.addEventListener("open", event => {
        if (!isCurrentConnection(socket, generation)) {
            return;
        }

        reconnectAttempts = 0;
        setConnectionStatus("connected", event);
        invokeCallback("onOpen", event, socket);
    });

    socket.addEventListener("message", event => {
        if (!isCurrentConnection(socket, generation)) {
            return;
        }

        handleIncomingMessage(event);
    });

    socket.addEventListener("close", event => {
        if (!isCurrentConnection(socket, generation)) {
            return;
        }

        state.connection.socket = null;
        setConnectionStatus("disconnected", event);
        invokeCallback("onClose", event);

        if (reconnectEnabled) {
            scheduleReconnect();
        }
    });

    socket.addEventListener("error", event => {
        if (!isCurrentConnection(socket, generation)) {
            return;
        }

        setConnectionStatus("error", event);
        invokeCallback("onError", event, socket);
    });

    return socket;
}

export function disconnectSocket({
    reconnect = false,
    code = 1000,
    reason = "Client disconnect"
} = {}) {
    reconnectEnabled = Boolean(reconnect);
    clearReconnectTimer();
    closeCurrentSocket({ code, reason });

    if (!reconnectEnabled) {
        setConnectionStatus("disconnected", {
            intentional: true,
            reason
        });
        return true;
    }

    scheduleReconnect();
    return true;
}

export function destroySocket() {
    reconnectEnabled = false;
    clearReconnectTimer();
    closeCurrentSocket({
        code: 1000,
        reason: "Socket module destroyed"
    });

    reconnectAttempts = 0;
    setConnectionStatus("idle");

    return true;
}


// ============================================================================
// MESSAGE SENDING
// ============================================================================

export function isSocketConnected() {
    return Boolean(
        typeof WebSocket !== "undefined" &&
        state.connection.socket &&
        state.connection.socket.readyState === WebSocket.OPEN
    );
}

export function getSocket() {
    return state.connection.socket;
}

export function getSocketStatus() {
    return connectionStatus;
}

export function sendSocketMessage(message) {
    const socket = state.connection.socket;

    if (
        typeof WebSocket === "undefined" ||
        !socket ||
        socket.readyState !== WebSocket.OPEN
    ) {
        return false;
    }

    let serializedMessage;

    try {
        serializedMessage = typeof message === "string"
            ? message
            : JSON.stringify(message);
    } catch (error) {
        console.error("Socket message could not be serialized:", error);
        return false;
    }

    if (typeof serializedMessage !== "string") {
        console.error("Socket message must be a string or a serializable value.");
        return false;
    }

    try {
        socket.send(serializedMessage);
        return true;
    } catch (error) {
        console.error("Socket message could not be sent:", error);
        invokeCallback("onError", error, socket);
        return false;
    }
}

export function sendSocketAction(action, payload = {}) {
    if (typeof action !== "string" || !action.trim()) {
        console.error("A non-empty socket action is required.");
        return false;
    }

    if (
        payload === null ||
        typeof payload !== "object" ||
        Array.isArray(payload)
    ) {
        console.error("Socket action payload must be an object.");
        return false;
    }

    return sendSocketMessage({
        ...payload,
        action: action.trim()
    });
}


// ============================================================================
// CALLBACK MANAGEMENT
// ============================================================================

export function setSocketHandlers(handlers = {}) {
    if (!handlers || typeof handlers !== "object" || Array.isArray(handlers)) {
        return false;
    }

    for (const [name, handler] of Object.entries(handlers)) {
        if (Object.prototype.hasOwnProperty.call(callbacks, name)) {
            callbacks[name] = isFunction(handler) ? handler : null;
        }
    }

    return true;
}
