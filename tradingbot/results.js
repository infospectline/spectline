import { state } from "./state.js";
import { timeCoord, priceCoord } from "./chart.js";


// ============================================================================
// MODULE STATE
// ============================================================================

const elements = {
    overlay: null,
    detailPanel: null,
    resultsContext: null,
    startingBalance: null,
    finalBalance: null,
    totalReturn: null,
    blockedTrades: null,
    winningTrades: null,
    losingTrades: null,
    maxWinStreak: null,
    maxLossStreak: null,
    winningPercentage: null,
    losingPercentage: null
};

let onTradeDetailRequest = null;


// ============================================================================
// INTERNAL HELPERS
// ============================================================================

function getElement(id) {
    return document.getElementById(id);
}

function getOverlay() {
    if (!elements.overlay) {
        elements.overlay = getElement("overlay");
    }

    return elements.overlay;
}

function getDetailPanel() {
    if (!elements.detailPanel) {
        elements.detailPanel = getElement("tradeDetailPanel");
    }

    return elements.detailPanel;
}

function setText(element, value) {
    if (element) {
        element.textContent = String(value);
    }
}

function toFiniteNumber(value, fallback = null) {
    if (
        value === null ||
        value === undefined ||
        value === ""
    ) {
        return fallback;
    }

    const number = Number(value);

    return Number.isFinite(number)
        ? number
        : fallback;
}

function fmtPrice(price) {
    const number = toFiniteNumber(price);

    return number === null
        ? "-"
        : number.toFixed(5);
}

function formatTime(timestamp) {
    const number = toFiniteNumber(timestamp);

    if (number === null) {
        return "-";
    }

    const date = new Date(number * 1000);

    if (Number.isNaN(date.getTime())) {
        return "-";
    }

    return date.toISOString()
        .replace("T", " ")
        .replace(".000Z", " UTC");
}

function formatNumber(value, digits = 5) {
    const number = toFiniteNumber(value);

    return number === null
        ? "-"
        : number.toFixed(digits);
}

function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, character => {
        const entities = {
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;"
        };

        return entities[character];
    });
}

function clearSelectedTradeSetup() {
    const overlay = getOverlay();

    if (!overlay) {
        return;
    }

    overlay.querySelectorAll(
        ".setup-line, .setup-point, .setup-label, .setup-level-highlight"
    ).forEach(element => element.remove());
}

function addSetupPoint(time, price, label, side = "right") {
    if (
        time === null ||
        time === undefined ||
        price === null ||
        price === undefined
    ) {
        return;
    }

    const overlay = getOverlay();

    if (!overlay) {
        return;
    }

    const x = timeCoord(time);
    const y = priceCoord(price);

    if (x === null || y === null) {
        return;
    }

    const point = document.createElement("div");
    point.className = "setup-point";
    point.style.left = `${x}px`;
    point.style.top = `${y}px`;
    point.style.zIndex = "30";
    overlay.appendChild(point);

    const labelElement = document.createElement("div");
    labelElement.className = "setup-label";
    labelElement.textContent = `${label} ${fmtPrice(price)}`;
    labelElement.style.left = `${x + (side === "left" ? -95 : 10)}px`;
    labelElement.style.top = `${y - 22}px`;
    overlay.appendChild(labelElement);
}

function addSetupLevel(price) {
    if (price === null || price === undefined) {
        return;
    }

    const overlay = getOverlay();

    if (!overlay) {
        return;
    }

    const y = priceCoord(price);

    if (y === null) {
        return;
    }

    const line = document.createElement("div");
    line.className = "setup-level-highlight";
    line.style.top = `${y - 1}px`;
    overlay.appendChild(line);

    const label = document.createElement("div");
    label.className = "setup-label";
    label.textContent = `MAIN LEVEL ${fmtPrice(price)}`;
    label.style.right = "10px";
    label.style.top = `${y - 22}px`;
    label.style.zIndex = "31";
    overlay.appendChild(label);
}


// ============================================================================
// MODULE INITIALIZATION
// ============================================================================

export function initializeResults({
    overlay = null,
    onTradeDetailRequest = null
} = {}) {
    elements.overlay = overlay || getElement("overlay");
    elements.detailPanel = getElement("tradeDetailPanel");
    elements.resultsContext = getElement("resultsContext");
    elements.startingBalance = getElement("startingBalance");
    elements.finalBalance = getElement("finalBalance");
    elements.totalReturn = getElement("totalReturn");
    elements.blockedTrades = getElement("blockedTrades");
    elements.winningTrades = getElement("winningTrades");
    elements.losingTrades = getElement("losingTrades");
    elements.maxWinStreak = getElement("maxWinStreak");
    elements.maxLossStreak = getElement("maxLossStreak");
    elements.winningPercentage = getElement("winningPercentage");
    elements.losingPercentage = getElement("losingPercentage");

    setTradeDetailRequestHandler(onTradeDetailRequest);

    return Boolean(elements.overlay && elements.detailPanel);
}

export function setTradeDetailRequestHandler(handler) {
    onTradeDetailRequest = typeof handler === "function"
        ? handler
        : null;
}


// ============================================================================
// RESULTS AND STATISTICS
// ============================================================================

export function updateTradeStats(trades = []) {
    const safeTrades = Array.isArray(trades) ? trades : [];

    const winning = safeTrades.filter(
        trade => trade?.result === "WIN"
    ).length;

    const losing = safeTrades.filter(
        trade => trade?.result === "LOSS"
    ).length;

    const total = winning + losing;

    let maxWinStreak = 0;
    let maxLossStreak = 0;
    let winStreak = 0;
    let lossStreak = 0;

    for (const trade of safeTrades) {
        if (trade?.result === "WIN") {
            winStreak++;
            lossStreak = 0;
            maxWinStreak = Math.max(maxWinStreak, winStreak);
        } else if (trade?.result === "LOSS") {
            lossStreak++;
            winStreak = 0;
            maxLossStreak = Math.max(maxLossStreak, lossStreak);
        }
    }

    const winningPercentage = total > 0
        ? (winning / total) * 100
        : 0;

    const losingPercentage = total > 0
        ? (losing / total) * 100
        : 0;

    setText(elements.winningTrades, winning);
    setText(elements.losingTrades, losing);
    setText(elements.maxWinStreak, maxWinStreak);
    setText(elements.maxLossStreak, maxLossStreak);
    setText(elements.winningPercentage, `${winningPercentage.toFixed(2)}%`);
    setText(elements.losingPercentage, `${losingPercentage.toFixed(2)}%`);

    return {
        winning,
        losing,
        total,
        maxWinStreak,
        maxLossStreak,
        winningPercentage,
        losingPercentage
    };
}

export function updateResultsContext(data = {}) {
    if (!elements.resultsContext) {
        elements.resultsContext = getElement("resultsContext");
    }

    if (!elements.resultsContext) {
        return;
    }

    const symbol = String(data?.symbol || "").toUpperCase();
    const timeframe = String(data?.timeframe || "").toUpperCase();

    let resultType = "TESTING";

    if (data?.results_source === "manual") {
        resultType = "MANUAL";
    } else if (data?.results_source === "testing") {
        resultType = "TESTING";
    }

    elements.resultsContext.textContent =
        `${symbol}, ${timeframe}, ${resultType}`;
}

export function updateResultSummary(data = {}) {
    const isManualResults = data?.results_source === "manual";
    const manualResults = data?.manual_analysis_results || {};

    const startingBalance = isManualResults
        ? toFiniteNumber(manualResults.starting_balance, 100000)
        : toFiniteNumber(data.starting_balance, 100000);

    const finalBalance = isManualResults
        ? toFiniteNumber(manualResults.final_balance, startingBalance)
        : toFiniteNumber(data.final_balance, 0);

    const totalReturn = isManualResults
        ? toFiniteNumber(manualResults.total_return, 0)
        : toFiniteNumber(data.total_return, 0);

    const blockedTrades = isManualResults
        ? 0
        : toFiniteNumber(data.blocked_trades, 0);

    const resultTrades =
        isManualResults && Array.isArray(manualResults.trades)
            ? manualResults.trades
            : (Array.isArray(data.trades) ? data.trades : []);

    setText(
        elements.startingBalance,
        `${startingBalance.toLocaleString("sk-SK")} €`
    );

    setText(
        elements.finalBalance,
        `${finalBalance.toLocaleString("sk-SK")} €`
    );

    setText(
        elements.totalReturn,
        `${totalReturn >= 0 ? "+" : ""}${totalReturn.toFixed(2)}%`
    );

    setText(elements.blockedTrades, blockedTrades);

    updateTradeStats(resultTrades);

    return resultTrades;
}


// ============================================================================
// TRADE DETAIL PANEL
// ============================================================================

export function renderTradeDetailPanel() {
    const panel = getDetailPanel();

    if (!panel) {
        return false;
    }

    const trade = state.results.selectedTradeDetail;

    if (!trade) {
        panel.classList.remove("visible");
        panel.innerHTML = "";
        return true;
    }

    const direction = trade.direction === "BUY"
        ? "LONG"
        : "SHORT";

    const result = trade.result || "OPEN";
    const pattern = trade.pattern || "-";
    const level = trade.setup_level ?? trade.level;
    const levelStrength = trade.level_strength ?? "-";
    const touches = trade.level_touches ?? "-";
    const swings = trade.setup_swings || {};

    let swingHtml = "";

    if (direction === "LONG") {
        swingHtml = `
            <div class="trade-detail-row">
                <span class="trade-detail-label">LOW 1</span>
                <span class="trade-detail-value">${formatNumber(swings.low1?.price)}</span>
            </div>
            <div class="trade-detail-row">
                <span class="trade-detail-label">LOW 2</span>
                <span class="trade-detail-value">${formatNumber(swings.low2?.price)}</span>
            </div>
            <div class="trade-detail-row">
                <span class="trade-detail-label">LOW2 > LOW1</span>
                <span class="trade-detail-value">${
                    swings.low1 && swings.low2 &&
                    Number(swings.low2.price) > Number(swings.low1.price)
                        ? "YES"
                        : "NO"
                }</span>
            </div>
        `;
    } else {
        swingHtml = `
            <div class="trade-detail-row">
                <span class="trade-detail-label">HIGH 1</span>
                <span class="trade-detail-value">${formatNumber(swings.high1?.price)}</span>
            </div>
            <div class="trade-detail-row">
                <span class="trade-detail-label">HIGH 2</span>
                <span class="trade-detail-value">${formatNumber(swings.high2?.price)}</span>
            </div>
            <div class="trade-detail-row">
                <span class="trade-detail-label">HIGH2 &lt; HIGH1</span>
                <span class="trade-detail-value">${
                    swings.high1 && swings.high2 &&
                    Number(swings.high2.price) < Number(swings.high1.price)
                        ? "YES"
                        : "NO"
                }</span>
            </div>
        `;
    }

    const tradeNumber = trade.trade_id ?? state.results.selectedTradeId;

    const protectedAtHtml = trade.protected
        ? `
            <div class="trade-detail-row">
                <span class="trade-detail-label">Protected at</span>
                <span class="trade-detail-value">${formatTime(trade.protected_at)}</span>
            </div>
        `
        : "";

    panel.innerHTML = `
        <div class="trade-detail-title">
            TRADE #${escapeHtml(tradeNumber)} | ${escapeHtml(direction)} | ${escapeHtml(result)}
        </div>

        <div class="trade-detail-section">
            <div class="trade-detail-section-title">ENTRY REASON</div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Pattern</span>
                <span class="trade-detail-value">${escapeHtml(pattern)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Main level</span>
                <span class="trade-detail-value">${formatNumber(level)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Level strength</span>
                <span class="trade-detail-value">${escapeHtml(levelStrength)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Touches</span>
                <span class="trade-detail-value">${escapeHtml(touches)}</span>
            </div>

            ${swingHtml}

            <div class="trade-detail-row">
                <span class="trade-detail-label">Breakout</span>
                <span class="trade-detail-value">${formatTime(trade.breakout_time)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Entry</span>
                <span class="trade-detail-value">${formatNumber(trade.entry)}</span>
            </div>
        </div>

        <div class="trade-detail-section">
            <div class="trade-detail-section-title">TRADE PLAN</div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Initial SL</span>
                <span class="trade-detail-value">${formatNumber(trade.initial_sl)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">TP</span>
                <span class="trade-detail-value">${formatNumber(trade.tp)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Risk</span>
                <span class="trade-detail-value">${formatNumber(trade.risk_distance)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">R:R</span>
                <span class="trade-detail-value">1:${escapeHtml(trade.reward_r_multiple ?? 3)}</span>
            </div>
        </div>

        <div class="trade-detail-section">
            <div class="trade-detail-section-title">33% PROTECTION</div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Trigger</span>
                <span class="trade-detail-value">${formatNumber(trade.protection_trigger)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Protection SL</span>
                <span class="trade-detail-value">${formatNumber(trade.protection_sl)}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Activated</span>
                <span class="trade-detail-value">${trade.protected ? "YES" : "NO"}</span>
            </div>

            ${protectedAtHtml}
        </div>

        <div class="trade-detail-section">
            <div class="trade-detail-section-title">SL BASIS</div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Type</span>
                <span class="trade-detail-value">${escapeHtml(trade.sl_basis?.type ?? "-")}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Reference</span>
                <span class="trade-detail-value">${escapeHtml(trade.sl_basis?.reference ?? "-")}</span>
            </div>

            <div class="trade-detail-row">
                <span class="trade-detail-label">Reference price</span>
                <span class="trade-detail-value">${formatNumber(trade.sl_basis?.reference_price)}</span>
            </div>
        </div>
    `;

    panel.classList.add("visible");
    return true;
}

export function setSelectedTradeDetail(tradeId, trade) {
    state.results.selectedTradeId = tradeId ?? null;
    state.results.selectedTradeDetail = trade ?? null;

    renderTradeDetailPanel();
    renderSelectedTradeSetup();
}

export function closeTradeDetail() {
    state.results.selectedTradeId = null;
    state.results.selectedTradeDetail = null;

    const panel = getDetailPanel();

    if (panel) {
        panel.classList.remove("visible");
        panel.innerHTML = "";
    }

    clearSelectedTradeSetup();

    const overlay = getOverlay();

    if (overlay) {
        overlay.querySelectorAll(".trade-box.selected")
            .forEach(element => element.classList.remove("selected"));
    }
}

export function renderSelectedTradeSetup() {
    clearSelectedTradeSetup();

    const trade = state.results.selectedTradeDetail;

    if (!trade) {
        return;
    }

    addSetupLevel(trade.setup_level);

    const swings = trade.setup_swings || {};

    if (
        trade.direction === "BUY" &&
        swings.low1 &&
        swings.low2
    ) {
        addSetupPoint(
            swings.low1.time,
            swings.low1.price,
            "LOW 1",
            "left"
        );

        addSetupPoint(
            swings.low2.time,
            swings.low2.price,
            "LOW 2"
        );
    }

    if (
        trade.direction === "SELL" &&
        swings.high1 &&
        swings.high2
    ) {
        addSetupPoint(
            swings.high1.time,
            swings.high1.price,
            "HIGH 1",
            "left"
        );

        addSetupPoint(
            swings.high2.time,
            swings.high2.price,
            "HIGH 2"
        );
    }

    if (trade.breakout_candle) {
        addSetupPoint(
            trade.breakout_candle.time,
            trade.breakout_candle.close,
            "BREAKOUT"
        );
    }
}


// ============================================================================
// HISTORICAL TRADE OVERLAY
// ============================================================================

function addMarketLevel(zone) {
    if (
        !zone ||
        zone.low === null ||
        zone.low === undefined ||
        zone.high === null ||
        zone.high === undefined
    ) {
        return;
    }

    const overlay = getOverlay();

    if (!overlay) {
        return;
    }

    const low = toFiniteNumber(zone.low);
    const high = toFiniteNumber(zone.high);

    if (low === null || high === null) {
        return;
    }

    const levelPrice = (low + high) / 2;
    const y = priceCoord(levelPrice);

    if (y === null) {
        return;
    }

    const element = document.createElement("div");
    element.className = "sr-zone";
    element.style.top = `${y}px`;
    element.style.height = "1px";

    const label = document.createElement("div");
    label.className = "sr-label";
    label.textContent =
        `S/R ${fmtPrice(levelPrice)} | ${zone.touches ?? 0} touches`;

    element.appendChild(label);
    overlay.appendChild(element);
}

function addTrade(trade, index, currentTime) {
    if (
        !trade ||
        trade.entry === null || trade.entry === undefined ||
        trade.sl === null || trade.sl === undefined ||
        trade.tp === null || trade.tp === undefined ||
        trade.opened_at === null || trade.opened_at === undefined
    ) {
        return false;
    }

    const entry = toFiniteNumber(trade.entry);
    const sl = toFiniteNumber(trade.sl);
    const tp = toFiniteNumber(trade.tp);
    const openedAt = toFiniteNumber(trade.opened_at);
    const closedAt = toFiniteNumber(
        trade.closed_at ?? currentTime ?? trade.opened_at
    );

    if (
        entry === null ||
        sl === null ||
        tp === null ||
        openedAt === null ||
        closedAt === null
    ) {
        return false;
    }

    const x1 = timeCoord(openedAt);
    const x2 = timeCoord(closedAt);
    const pEntry = priceCoord(entry);
    const pSl = priceCoord(sl);
    const pTp = priceCoord(tp);

    if (
        x1 === null ||
        x2 === null ||
        pEntry === null ||
        pSl === null ||
        pTp === null
    ) {
        return false;
    }

    const overlay = getOverlay();

    if (!overlay) {
        return false;
    }

    const left = Math.min(x1, x2);
    const right = Math.max(x1, x2);
    const top = Math.min(pSl, pTp);
    const bottom = Math.max(pSl, pTp);

    const box = document.createElement("div");

    const resultClass = trade.result === "WIN"
        ? "win"
        : trade.result === "LOSS"
            ? "loss"
            : "open";

    box.className = `trade-box ${resultClass}`;

    const fallbackTradeId = index + 1;
    const tradeId = toFiniteNumber(trade.trade_id, fallbackTradeId);

    box.dataset.tradeId = String(tradeId);

    if (
        state.results.selectedTradeId !== null &&
        Number(state.results.selectedTradeId) === tradeId
    ) {
        box.classList.add("selected");
    }

    box.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();

        state.results.selectedTradeId = tradeId;

        overlay.querySelectorAll(".trade-box.selected")
            .forEach(element => element.classList.remove("selected"));

        box.classList.add("selected");

        if (onTradeDetailRequest) {
            onTradeDetailRequest(tradeId);
        }
    });

    box.style.left = `${left}px`;
    box.style.top = `${top}px`;
    box.style.width = `${Math.max(22, right - left)}px`;
    box.style.height = `${Math.max(28, bottom - top)}px`;

    const title = document.createElement("div");
    title.className = "trade-title";
    title.textContent =
        `${index + 1}. ${trade.direction || ""} ${trade.result || "OPEN"}`;

    box.appendChild(title);

    const points = [
        ["entry", pEntry, entry],
        ["tp", pTp, tp],
        ["sl", pSl, sl]
    ];

    for (const [kind, absoluteY, price] of points) {
        const line = document.createElement("div");
        line.className = `trade-line ${kind}`;
        line.style.top = `${absoluteY - top}px`;
        box.appendChild(line);

        const label = document.createElement("div");
        label.className = `trade-price ${kind}`;
        label.textContent = `${kind.toUpperCase()} ${fmtPrice(price)}`;

        const localY = Math.max(
            0,
            Math.min(bottom - top - 12, absoluteY - top - 6)
        );

        label.style.top = `${localY}px`;
        box.appendChild(label);
    }

    overlay.appendChild(box);
    return true;
}

export function renderResultsOverlay(data = state.chart.lastData, {
    showLevels = false,
    showWinningTrades = false,
    showLosingTrades = false
} = {}) {
    const overlay = getOverlay();

    if (!overlay) {
        return false;
    }

    overlay.querySelectorAll(".trade-box, .sr-zone")
        .forEach(element => element.remove());

    clearSelectedTradeSetup();

    if (!data?.ready) {
        return false;
    }

    if (showLevels && Array.isArray(data.levels)) {
        data.levels.forEach(addMarketLevel);
    }

    const chart = state.chart.instance;
    const visibleRange = chart?.timeScale().getVisibleRange();

    if (visibleRange) {
        const fromTime = Number(visibleRange.from);
        const toTime = Number(visibleRange.to);
        const trades = Array.isArray(data.trades) ? data.trades : [];

        trades.forEach((trade, index) => {
            if (
                !trade ||
                trade.opened_at === null ||
                trade.opened_at === undefined
            ) {
                return;
            }

            const opened = toFiniteNumber(trade.opened_at);

            const closed = toFiniteNumber(
                trade.closed_at ?? data.current_time ?? trade.opened_at
            );

            if (opened === null || closed === null) {
                return;
            }

            if (closed < fromTime || opened > toTime) {
                return;
            }

            if (
                trade.result === "WIN" &&
                showWinningTrades
            ) {
                addTrade(trade, index, data.current_time);
            }

            if (
                trade.result === "LOSS" &&
                showLosingTrades
            ) {
                addTrade(trade, index, data.current_time);
            }

            if (!trade.result) {
                addTrade(trade, index, data.current_time);
            }
        });

        if (data.trade) {
            const alreadyClosed = trades.some(trade =>
                trade &&
                trade.opened_at === data.trade.opened_at &&
                trade.entry === data.trade.entry
            );

            if (!alreadyClosed) {
                addTrade(data.trade, trades.length, data.current_time);
            }
        }
    }

    if (state.results.selectedTradeDetail) {
        renderSelectedTradeSetup();
    }

    return true;
}