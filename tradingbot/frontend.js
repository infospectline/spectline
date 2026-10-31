let WEB_MODE = false;

const tradingbotLaunchTicket =
    new URLSearchParams(
        window.location.hash.slice(1)
    ).get("ticket");

if (tradingbotLaunchTicket) {
    window.history.replaceState(
        null,
        "",
        window.location.pathname + window.location.search
    );
}

async function loadWebMode() {
    try {
        const response = await fetch("/api/config");
        const config = await response.json();

        WEB_MODE = Boolean(config.web_mode);
        const logoutButton = document.getElementById("logoutButton");

        if (logoutButton) {
            logoutButton.style.display = WEB_MODE ? "block" : "none";
        }
        
        applyWebTimeframeRestrictions();

    } catch (error) {
        console.error("Failed to load web mode:", error);
    }
}

const WEB_ALLOWED_TIMEFRAMES = [
    "M15",
    "M30",
    "H1",
    "H4",
    "D1",
    "W1"
];

function applyWebTimeframeRestrictions() {
    if (!WEB_MODE) {
        return;
    }

    Array.from(timeframeSelect.options).forEach(option => {
        if (!WEB_ALLOWED_TIMEFRAMES.includes(option.value)) {
            option.remove();
        }
    });

    if (
        !WEB_ALLOWED_TIMEFRAMES.includes(
            timeframeSelect.value
        )
    ) {
        timeframeSelect.value = "M15";
    }
}

const chartContainer = document.getElementById("chart");
    const overlay = document.getElementById("overlay");
    const chartPanel = document.getElementById("chart-panel");

    function isMobileTouchDevice() {
        return window.matchMedia(
            "(max-width: 900px) and (pointer: coarse)"
        ).matches;
    }

    const trainingButton =
        document.getElementById("trainingButton");

    const testingButton =
        document.getElementById("testingButton");

    const learningButton =
        document.getElementById("learningButton");

    const updateDataButton =
        document.getElementById("updateDataButton");

    const resultsTabButton =
        document.getElementById("resultsTabButton");

    const learningTabButton =
        document.getElementById("learningTabButton");

    const deleteAnchorButton =
        document.getElementById(
            "deleteAnchorButton"
        );

    const expandAnchorButton =
        document.getElementById(
            "expandAnchorButton"
        );

    const expandTradeSetupButton =
        document.getElementById(
            "expandTradeSetupButton"
        );

    const deleteTradeSetupButton =
        document.getElementById(
            "deleteTradeSetupButton"
        );

    const mt5OfflinePopup =
        document.getElementById("mt5-offline-popup");

    let mt5OfflinePopupTimer = null;

    function showMT5OfflinePopup() {
        if (!mt5OfflinePopup) {
            return;
        }

        mt5OfflinePopup.classList.add("show");

        clearTimeout(mt5OfflinePopupTimer);

        mt5OfflinePopupTimer = setTimeout(() => {
            mt5OfflinePopup.classList.remove("show");
        }, 5000);
    }

    const resultsView = 
        document.getElementById("resultsView");

    const learningView = 
        document.getElementById("learningView");

    const resultsContext = 
        document.getElementById("resultsContext");

    const learningMainView = 
        document.getElementById("learningMainView");

    function showResultsTab() {
        resultsView.style.display = "block";
        learningView.style.display = "none";
    }

    function showLearningTab() {
        resultsView.style.display = "none";
        learningView.style.display = "block";
    }

    const learningMarket = 
        document.getElementById("learningMarket");

    const learningTimeframe = 
        document.getElementById("learningTimeframe");

    const learningTradeCount = 
        document.getElementById("learningTradeCount");

    const learningAddButton = 
        document.getElementById("learningAddButton");

    const manualAnalysisButton =    
        document.getElementById("manualAnalysisButton");

    let manualAnalysisActive = false;

    const addZoneButton =
        document.getElementById("addZoneButton");

    const zoneTypeMenu =
        document.getElementById("zoneTypeMenu");

    const addSupportButton =
        document.getElementById("addSupportButton");

    const addResistanceButton =
        document.getElementById("addResistanceButton");

    const learningZoneList =
        document.getElementById("learningZoneList");

    const learningListView =
        document.getElementById("learningListView");

    const learningDetailView =
        document.getElementById("learningDetailView");

    const learningTradeList =
        document.getElementById("learningTradeList");

    const learningSaveButton =
        document.getElementById("learningSaveButton");

    const learningDeleteButton =
        document.getElementById("learningDeleteButton");

    const learningAnchorTime =
        document.getElementById("learningAnchorTime");

    const learningAnchorPrice =
        document.getElementById("learningAnchorPrice");

    const learningRRModeButtons =
        document.querySelectorAll(".learning-rr-mode-button");

    const learningAnchorPicker =
        document.getElementById("learningAnchorPicker");

    const learningDetailTradeNumber =
        document.getElementById("learningDetailTradeNumber");

    const learningSetupTimeframe =
        document.getElementById("learningSetupTimeframe");

    const learningEntryTimeframe =
        document.getElementById("learningEntryTimeframe");

    const learningBreakout =
        document.getElementById("learningBreakout");

    const learningEntryCandle =
        document.getElementById("learningEntryCandle");

    const learningEntry =
        document.getElementById("learningEntry");

    const learningSL =
        document.getElementById("learningSL");

    const learningTP =
        document.getElementById("learningTP");

    const showLevels =
        document.getElementById("showLevels");

    const showWinningTrades =
        document.getElementById("showWinningTrades");

    const showLosingTrades =
        document.getElementById("showLosingTrades");

    const showManualSetups =
        document.getElementById("showManualSetups");

    const chart = LightweightCharts.createChart(chartContainer, {
        layout: {
            background: { color: "#111" },
            textColor: "#bbb"
        },

        localization: {
            timeFormatter: (time) => {
                const date = new Date(Number(time) * 1000);
                return date.toISOString().replace("T", " ").replace(".000Z", " UTC");
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
            mode: LightweightCharts.CrosshairMode.Normal
        },
        timeScale: {
            timeVisible: true,
            secondsVisible: false,
            rightOffset: 5,
            tickMarkFormatter: (time) => {
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

    const candleSeries = chart.addCandlestickSeries();

    let lastData = null;
    let socket = null;
    let viewportTimer = null;
    let chartDataLoaded = false;
    let activeSessionId = null;
    let pendingSessionId = null;

    let selectedTradeId = null;
    let selectedTradeDetail = null;

    let chartMouseDownX = 0;
    let chartMouseDownY = 0;
    let chartWasDragged = false;

    let learningMode = false;
    let learningTrades = [];
    let selectedLearningTradeId = null;
    let nextLearningTradeId = 1;

    let selectedLearningTradeIsNew = false;
    let activePriceTarget = null;
    let activeAnchorPicker = false;
    let pendingAnchorCenter = false;

    function updateLoading(data) {
        const screen = document.getElementById("loading-screen");
        const bar = document.getElementById("loading-bar");
        const percent = document.getElementById("loading-percent");
        const message = document.getElementById("loading-message");
        const title = document.getElementById("loading-title");

        const progress = Math.max(
            0,
            Math.min(100, Number(data?.startup_progress ?? 0))
        );

        const symbol = data?.symbol || marketSelect.value || "";

        title.textContent = symbol
            ? `Loading ${symbol} emulator`
            : "Loading emulator";

        bar.style.width = `${progress}%`;
        percent.textContent = `${progress}%`;
        message.textContent = data?.startup_message || "Loading...";

        const isLoading =
            data?.startup_stage === "Loading market data" ||
            data?.startup_stage === "Starting" ||
            data?.loading === true;

       if (!isLoading && data?.ready && data?.h1?.length) {
            screen.style.display = "none";
        } else {
            screen.style.display = "flex";
        }
    }

    trainingButton.addEventListener("click", () => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            document.getElementById("status").textContent =
                "Backend is not connected.";
            return;
        }

        if (!lastData || !lastData.ready) {
            document.getElementById("status").textContent =
                "Chart is not ready.";
            return;
        }

        trainingButton.disabled = true;
        testingButton.disabled = true;

        socket.send(JSON.stringify({
            action: "training"
        }));

        document.getElementById("status").textContent =
            `Training ${marketSelect.value} ${timeframeSelect.value}...`;
    });


    testingButton.addEventListener("click", () => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            document.getElementById("status").textContent =
                "Backend is not connected.";
            return;
        }

        if (!lastData || !lastData.ready) {
            document.getElementById("status").textContent =
                "Chart is not ready.";
            return;
        }

        testingButton.disabled = true;
        trainingButton.disabled = true;

        socket.send(JSON.stringify({
            action: "testing"
        }));

        document.getElementById("status").textContent =
            `Testing ${marketSelect.value} ${timeframeSelect.value}...`;
    });

    function updateResults(trades) {
        const winning = trades.filter(t => t.result === "WIN").length;
        const losing = trades.filter(t => t.result === "LOSS").length;
        const total = winning + losing;

        let maxWinStreak = 0;
        let maxLossStreak = 0;
        let winStreak = 0;
        let lossStreak = 0;

        for (const trade of trades) {
            if (trade.result === "WIN") {
                winStreak++;
                lossStreak = 0;
                maxWinStreak = Math.max(maxWinStreak, winStreak);
            } else if (trade.result === "LOSS") {
                lossStreak++;
                winStreak = 0;
                maxLossStreak = Math.max(maxLossStreak, lossStreak);
            }
        }

        const winningPercentage = total > 0 ? (winning / total) * 100 : 0;
        const losingPercentage = total > 0 ? (losing / total) * 100 : 0;

        document.getElementById("winningTrades").textContent = winning;
        document.getElementById("losingTrades").textContent = losing;
        document.getElementById("maxWinStreak").textContent = maxWinStreak;
        document.getElementById("maxLossStreak").textContent = maxLossStreak;
        document.getElementById("winningPercentage").textContent = `${winningPercentage.toFixed(2)}%`;
        document.getElementById("losingPercentage").textContent = `${losingPercentage.toFixed(2)}%`;
    }

    function refreshOverlay() {
        chart.applyOptions({
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight
        });

        requestAnimationFrame(renderOverlay);
    }

    showLevels.addEventListener("click", () => {
        showLevels.classList.toggle("active");
        refreshOverlay();
    });

    showWinningTrades.addEventListener("click", () => {
        showWinningTrades.classList.toggle("active");
        refreshOverlay();
    });

    showLosingTrades.addEventListener("click", () => {
        showLosingTrades.classList.toggle("active");
        refreshOverlay();
    });

    showManualSetups.addEventListener("click", () => {
        showManualSetups.classList.toggle("active");
        refreshOverlay();
    });

    function fmtPrice(price) {
        return Number(price).toFixed(5);
    }

    function timeCoord(time) {
        return chart.timeScale().timeToCoordinate(Number(time));
    }

    function priceCoord(price) {
        return candleSeries.priceToCoordinate(Number(price));
    }

    function clearOverlay() {
        overlay.innerHTML = "";
    }

        
    function addZone(zone, index) {
        if (!zone || zone.low == null || zone.high == null) return;

        const levelPrice =
            (Number(zone.low) + Number(zone.high)) / 2;

        const y = priceCoord(levelPrice);

        if (y == null) return;

        const el = document.createElement("div");
        el.className = "sr-zone";

        el.style.top = `${y}px`;
        el.style.height = "1px";

        const label = document.createElement("div");
        label.className = "sr-label";
        label.textContent =
            `S/R ${fmtPrice(levelPrice)} | ${zone.touches ?? 0} touches`;

        el.appendChild(label);
        overlay.appendChild(el);
    }


    function clearSelectedTradeSetup() {
        overlay.querySelectorAll(
            ".setup-line, .setup-point, .setup-label, .setup-level-highlight"
        ).forEach(el => el.remove());
    }

    function closeTradeDetail() {
        selectedTradeId = null;
        selectedTradeDetail = null;

        const panel = document.getElementById("tradeDetailPanel");
        panel.classList.remove("visible");
        panel.innerHTML = "";

        clearSelectedTradeSetup();

        overlay.querySelectorAll(".trade-box.selected")
            .forEach(el => el.classList.remove("selected"));
    }

    function formatTime(timestamp) {
        if (timestamp == null) return "-";

        const date = new Date(Number(timestamp) * 1000);

        return date.toISOString()
            .replace("T", " ")
            .replace(".000Z", " UTC");
    }

    function formatNumber(value, digits = 5) {
        if (value == null || !Number.isFinite(Number(value))) {
            return "-";
        }

        return Number(value).toFixed(digits);
    }

    function addSetupPoint(time, price, label, side = "right") {
        if (time == null || price == null) return;

        const x = timeCoord(time);
        const y = priceCoord(price);

        if (x == null || y == null) return;

        const point = document.createElement("div");
        point.className = "setup-point";

        point.style.left = `${x}px`;
        point.style.top = `${y}px`;
        point.style.zIndex = "30";

        overlay.appendChild(point);

        const labelEl = document.createElement("div");
        labelEl.className = "setup-label";
        labelEl.textContent = `${label} ${fmtPrice(price)}`;

        labelEl.style.left =
            `${x + (side === "left" ? -95 : 10)}px`;

        labelEl.style.top = `${y - 22}px`;

        overlay.appendChild(labelEl);
    }

    function addSetupLevel(price) {
        if (price == null) return;

        const y = priceCoord(price);

        if (y == null) return;

        const line = document.createElement("div");
        line.className = "setup-level-highlight";

        line.style.top = `${y - 1}px`;

        overlay.appendChild(line);

        const label = document.createElement("div");
        label.className = "setup-label";

        label.textContent =
            `MAIN LEVEL ${fmtPrice(price)}`;

        label.style.right = "10px";
        label.style.top = `${y - 22}px`;
        label.style.zIndex = "31";

        overlay.appendChild(label);
    }

    function renderSelectedTradeSetup() {
        clearSelectedTradeSetup();

        if (!selectedTradeDetail) {
            return;
        }

        const trade = selectedTradeDetail;

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

        const breakoutCandle =
            trade.breakout_candle;

        if (breakoutCandle) {
            addSetupPoint(
                breakoutCandle.time,
                breakoutCandle.close,
                "BREAKOUT"
            );
        }
    }

    function renderTradeDetailPanel() {
        const panel =
            document.getElementById("tradeDetailPanel");

        if (!selectedTradeDetail) {
            panel.classList.remove("visible");
            panel.innerHTML = "";
            return;
        }

        const trade = selectedTradeDetail;

        const direction =
            trade.direction === "BUY"
                ? "LONG"
                : "SHORT";

        const result =
            trade.result || "OPEN";

        const pattern =
            trade.pattern || "-";

        const level =
            trade.setup_level ?? trade.level;

        const levelStrength =
            trade.level_strength;

        const touches =
            trade.level_touches;

        const swings = trade.setup_swings || {};

        let swingHtml = "";

        if (direction === "LONG") {
            swingHtml = `
                <div class="trade-detail-row">
                    <span class="trade-detail-label">LOW 1</span>
                    <span class="trade-detail-value">
                        ${formatNumber(swings.low1?.price)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">LOW 2</span>
                    <span class="trade-detail-value">
                        ${formatNumber(swings.low2?.price)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">LOW2 > LOW1</span>
                    <span class="trade-detail-value">
                        ${
                            swings.low1 &&
                            swings.low2 &&
                            Number(swings.low2.price) >
                            Number(swings.low1.price)
                                ? "YES"
                                : "NO"
                        }
                    </span>
                </div>
            `;
        } else {
            swingHtml = `
                <div class="trade-detail-row">
                    <span class="trade-detail-label">HIGH 1</span>
                    <span class="trade-detail-value">
                        ${formatNumber(swings.high1?.price)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">HIGH 2</span>
                    <span class="trade-detail-value">
                        ${formatNumber(swings.high2?.price)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">HIGH2 < HIGH1</span>
                    <span class="trade-detail-value">
                        ${
                            swings.high1 &&
                            swings.high2 &&
                            Number(swings.high2.price) <
                            Number(swings.high1.price)
                                ? "YES"
                                : "NO"
                        }
                    </span>
                </div>
            `;
        }

        panel.innerHTML = `
            <div class="trade-detail-title">
                TRADE #${trade.trade_id ?? selectedTradeId}
                | ${direction}
                | ${result}
            </div>

            <div class="trade-detail-section">
                <div class="trade-detail-section-title">
                    ENTRY REASON
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Pattern</span>
                    <span class="trade-detail-value">
                        ${pattern}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Main level</span>
                    <span class="trade-detail-value">
                        ${formatNumber(level)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Level strength</span>
                    <span class="trade-detail-value">
                        ${levelStrength ?? "-"}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Touches</span>
                    <span class="trade-detail-value">
                        ${touches ?? "-"}
                    </span>
                </div>

                ${swingHtml}

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Breakout</span>
                    <span class="trade-detail-value">
                        ${formatTime(trade.breakout_time)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Entry</span>
                    <span class="trade-detail-value">
                        ${formatNumber(trade.entry)}
                    </span>
                </div>
            </div>

            <div class="trade-detail-section">
                <div class="trade-detail-section-title">
                    TRADE PLAN
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Initial SL</span>
                    <span class="trade-detail-value">
                        ${formatNumber(trade.initial_sl)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">TP</span>
                    <span class="trade-detail-value">
                        ${formatNumber(trade.tp)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Risk</span>
                    <span class="trade-detail-value">
                        ${formatNumber(trade.risk_distance)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">R:R</span>
                    <span class="trade-detail-value">
                        1:${trade.reward_r_multiple ?? 3}
                    </span>
                </div>
            </div>

            <div class="trade-detail-section">
                <div class="trade-detail-section-title">
                    33% PROTECTION
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Trigger</span>
                    <span class="trade-detail-value">
                        ${formatNumber(trade.protection_trigger)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Protection SL</span>
                    <span class="trade-detail-value">
                        ${formatNumber(trade.protection_sl)}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Activated</span>
                    <span class="trade-detail-value">
                        ${
                            trade.protected
                                ? "YES"
                                : "NO"
                        }
                    </span>
                </div>

                ${
                    trade.protected
                        ? `
                            <div class="trade-detail-row">
                                <span class="trade-detail-label">Protected at</span>
                                <span class="trade-detail-value">
                                    ${formatTime(trade.protected_at)}
                                </span>
                            </div>
                        `
                        : ""
                }
            </div>

            <div class="trade-detail-section">
                <div class="trade-detail-section-title">
                    SL BASIS
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Type</span>
                    <span class="trade-detail-value">
                        ${trade.sl_basis?.type ?? "-"}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Reference</span>
                    <span class="trade-detail-value">
                        ${trade.sl_basis?.reference ?? "-"}
                    </span>
                </div>

                <div class="trade-detail-row">
                    <span class="trade-detail-label">Reference price</span>
                    <span class="trade-detail-value">
                        ${formatNumber(
                            trade.sl_basis?.reference_price
                        )}
                    </span>
                </div>
            </div>
        `;

        panel.classList.add("visible");
    }

    function addTrade(trade, index) {
        if (!trade || trade.entry == null || trade.sl == null || trade.tp == null) return;
        if (trade.opened_at == null) return;

        const x1 = timeCoord(trade.opened_at);
        const x2 = timeCoord(trade.closed_at ?? lastData?.current_time ?? trade.opened_at);
        if (x1 == null || x2 == null) return;

        const pEntry = priceCoord(trade.entry);
        const pSl = priceCoord(trade.sl);
        const pTp = priceCoord(trade.tp);
        if (pEntry == null || pSl == null || pTp == null) return;

        const left = Math.min(x1, x2);
        const right = Math.max(x1, x2);
        const top = Math.min(pSl, pTp);
        const bottom = Math.max(pSl, pTp);

        const box = document.createElement("div");
        const resultClass = trade.result === "WIN" ? "win" : trade.result === "LOSS" ? "loss" : "open";
        box.className = `trade-box ${resultClass}`;

        const tradeId = Number(
            trade.trade_id ?? (index + 1)
        );

        box.dataset.tradeId = String(tradeId);

        if (
            selectedTradeId !== null &&
            Number(selectedTradeId) === tradeId
        ) {
            box.classList.add("selected");
        }

        box.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();

            selectedTradeId = tradeId;

            overlay.querySelectorAll(".trade-box.selected")
                .forEach(el => el.classList.remove("selected"));

            box.classList.add("selected");

            if (!socket || socket.readyState !== WebSocket.OPEN) {
                return;
            }

            socket.send(JSON.stringify({
                action: "trade_detail",
                trade_id: tradeId
            }));
        });

        box.style.left = `${left}px`;
        box.style.top = `${top}px`;
        box.style.width = `${Math.max(22, right - left)}px`;
        box.style.height = `${Math.max(28, bottom - top)}px`;

        const title = document.createElement("div");
        title.className = "trade-title";
        title.textContent = `${index + 1}. ${trade.direction} ${trade.result || "OPEN"}`;
        box.appendChild(title);

        const points = [
            ["entry", pEntry, trade.entry],
            ["tp", pTp, trade.tp],
            ["sl", pSl, trade.sl]
        ];

        for (const [kind, absoluteY, price] of points) {
            const line = document.createElement("div");
            line.className = `trade-line ${kind}`;
            line.style.top = `${absoluteY - top}px`;
            box.appendChild(line);

            const label = document.createElement("div");
            label.className = `trade-price ${kind}`;
            label.textContent = `${kind.toUpperCase()} ${fmtPrice(price)}`;

            const localY = Math.max(0, Math.min(bottom - top - 12, absoluteY - top - 6));
            label.style.top = `${localY}px`;
            box.appendChild(label);
        }

        overlay.appendChild(box);
    }

    function renderManualSetupOverlay(trade, index) {
        const geometry =
            trade?.trade_geometry;

        if (!geometry) {
            return;
        }

        if (
            geometry.start_time == null ||
            geometry.end_time == null
        ) {
            return;
        }

        const x1 =
            geometryTimeToX(
                geometry.start_time
            );

        const x2 =
            geometryTimeToX(
                geometry.end_time
            );

        if (
            x1 == null ||
            x2 == null
        ) {
            return;
        }

        const prices = [
            geometry.breakout?.price,
            geometry.entry?.price,
            geometry.sl?.price,
            geometry.tp?.price
        ]
            .filter(
                price =>
                    price != null &&
                    Number.isFinite(
                        Number(price)
                    )
            )
            .map(Number);

        if (!prices.length) {
            return;
        }

        const high =
            Math.max(...prices);

        const low =
            Math.min(...prices);

        const y1 =
            geometryPriceToY(high);

        const y2 =
            geometryPriceToY(low);

        if (
            y1 == null ||
            y2 == null
        ) {
            return;
        }

        const left =
            Math.min(x1, x2);

        const right =
            Math.max(x1, x2);

        const top =
            Math.min(y1, y2);

        const bottom =
            Math.max(y1, y2);

        const width =
            Math.max(
                12,
                right - left
            );

        const height =
            Math.max(
                20,
                bottom - top
            );

        const rectangle =
            document.createElement("div");

        rectangle.className =
"learning-geometry-rect learning-trade-setup-rect manual-setup-view-only";

        rectangle.style.left =
            `${left}px`;

        rectangle.style.top =
            `${top}px`;

        rectangle.style.width =
            `${width}px`;

        rectangle.style.height =
            `${height}px`;

        rectangle.title =
            `MANUAL SETUP ${index + 1}`;

        overlay.appendChild(
            rectangle
        );

        const entryY = geometry.entry?.price != null
            ? geometryPriceToY(geometry.entry.price)
            : null;

        function addSetupArea(targetPrice, className) {
            if (entryY == null || targetPrice == null) {
                return;
            }

            const targetY = geometryPriceToY(targetPrice);

            if (targetY == null) {
                return;
            }

            const area = document.createElement("div");
            area.className = `learning-setup-area ${className} manual-setup-view-only`;

            area.style.left = `${left}px`;
            area.style.top = `${Math.min(entryY, targetY)}px`;
            area.style.width = `${width}px`;
            area.style.height = `${Math.max(1, Math.abs(targetY - entryY))}px`;

            overlay.appendChild(area);
        }

        addSetupArea(
            geometry.tp?.price,
            "learning-setup-profit-area"
        );

        addSetupArea(
            geometry.sl?.price,
            "learning-setup-risk-area"
        );

        const levels = [
            {
                key: "breakout",
                label: "BREAKOUT",
                className:
                    "learning-setup-breakout"
            },
            {
                key: "entry",
                label: "ENTRY",
                className:
                    "learning-setup-entry"
            },
            {
                key: "sl",
                label: "SL",
                className:
                    "learning-setup-sl"
            },
            {
                key: "tp",
                label: "TP",
                className:
                    "learning-setup-tp"
            }
        ];

        levels.forEach(level => {
            const point =
                geometry[level.key];

            if (
                !point ||
                point.price == null
            ) {
                return;
            }

            const y =
                geometryPriceToY(
                    point.price
                );

            if (y == null) {
                return;
            }

            const line =
                document.createElement("div");

            line.className =
                `learning-setup-level ${level.className} manual-setup-view-only`;

            line.style.left =
                `${left}px`;

            line.style.width =
                `${width}px`;

            line.style.top =
                `${y - 2}px`;

            line.title =
                `${level.label} ${fmtPrice(point.price)}`;

            overlay.appendChild(
                line
            );

            const labelElement =
                document.createElement("div");

            labelElement.className =
                "learning-setup-label manual-setup-view-only";

            labelElement.textContent =
                `${level.label} ${fmtPrice(point.price)} | ${formatGeometryTime(point.time)}`;

            labelElement.style.left =
                `${left + 6}px`;

            labelElement.style.top =
                `${y - 18}px`;

            overlay.appendChild(
                labelElement
            );
        });
    }

    function renderOverlay() {
        if (!lastData || !lastData.ready) return;

        clearOverlay();

        if (showManualSetups.classList.contains("active")) {
            learningTrades.forEach(
                (trade, index) => {
                    renderManualSetupOverlay(
                        trade,
                        index
                    );
                }
            );

            return;
        }

        if (showLevels.classList.contains("active")) {
            const levels = lastData.levels || [];
            levels.forEach(addZone);
        }

        const visibleRange = chart.timeScale().getVisibleRange();

        if (!visibleRange) return;

        const fromTime = Number(visibleRange.from);
        const toTime = Number(visibleRange.to);

        const trades = lastData.trades || [];

        trades.forEach((trade, index) => {
            if (!trade || trade.opened_at == null) return;

            const opened = Number(trade.opened_at);
            const closed = Number(
                trade.closed_at ??
                lastData.current_time ??
                trade.opened_at
            );

            if (closed < fromTime || opened > toTime) {
                return;
            }

            if (
                trade.result === "WIN" &&
                showWinningTrades.classList.contains("active")
            ) {
                addTrade(trade, index);
            }

            if (
                trade.result === "LOSS" &&
                showLosingTrades.classList.contains("active")
            ) {
                addTrade(trade, index);
            }

            if (!trade.result) {
                addTrade(trade, index);
            }
        });

        if (lastData.trade) {
            const alreadyClosed = trades.some(t =>
                t.opened_at === lastData.trade.opened_at &&
                t.entry === lastData.trade.entry
            );

            if (!alreadyClosed) {
                addTrade(lastData.trade, trades.length);
            }
        }

        if (
            learningMode &&
            learningDetailView.style.display !== "none" &&
            selectedLearningTradeId !== null
        ) {
            renderLearningAnchor();
            renderLearningGeometry();
        }

        if (selectedTradeDetail) {
            renderSelectedTradeSetup();
        }

    }

    function syncSelectorsFromBackend(data) {
        const symbol = data?.symbol;
        const timeframe = data?.timeframe;

        if (symbol && marketSelect.value !== symbol) {
            const optionExists = Array.from(marketSelect.options)
                .some(option => option.value === symbol);

            if (optionExists) {
                marketSelect.value = symbol;
            }
        }

        if (timeframe && timeframeSelect.value !== timeframe) {
            const optionExists = Array.from(timeframeSelect.options)
                .some(option => option.value === timeframe);

            if (optionExists) {
                timeframeSelect.value = timeframe;
            }
        }
    }

    function resetChartView() {
        chartDataLoaded = false;
        lastData = null;

        closeTradeDetail();

        clearOverlay();

        candleSeries.setData([]);
    }

    function getStatusText(data) {
        const symbol = data?.symbol || marketSelect.value || "";
        const timeframe = data?.timeframe || timeframeSelect.value || "";
        const price = Number(data?.current_price ?? 0);
        const levels = (data?.levels || []).length;
        const trades = (data?.trades || []).length;
        const state = data?.analysis_status || "idle";

        let status = "Waiting for analysis";

        if (data?.startup_stage === "Loading market data") {
            status = "Loading market data";
        } else if (state === "training") {
            status = "TRAINING running";
        } else if (state === "testing") {
            status = "TESTING running";
        } else if (data?.analysis_phase === "training_completed") {
            status = "TRAINING completed";
        } else if (data?.analysis_phase === "testing_completed") {
            status = "TESTING completed";
        } else if (data?.analysis_phase === "training_error") {
            status = "TRAINING error";
        } else if (data?.analysis_phase === "testing_error") {
            status = "TESTING error";
        } else if (state === "manual_analysis") {
            status = "MANUAL ANALYSIS running";
        } else if (data?.analysis_phase === "manual_analysis_completed") {
            status = "MANUAL ANALYSIS completed";
        } else if (data?.analysis_phase === "manual_analysis_error") {
            status = "MANUAL ANALYSIS error";
        } else if (state === "running") {
            status = "Analysis running";
        } else if (state === "completed") {
            status = "Analysis completed";
        } else if (state === "paused") {
            status = "Analysis paused";
        }

        return `${symbol} | ${timeframe} | ${status} | Price ${price.toFixed(5)} | Levels ${levels} | Trades ${trades}`;
    }

    function updateManualAnalysisButton(data) {

        if (!manualAnalysisButton) {
            return;
        }

        if (!manualAnalysisActive) {
            return;
        }

        if (data?.analysis_status === "manual_analysis") {

            const progress = Math.max(
                0,
                Math.min(
                    100,
                    Number(
                        data?.manual_analysis_progress ?? 0
                    )
                )
            );

            manualAnalysisButton.textContent =
                `ANALYSIS ${progress}%`;

            return;
        }

        if (
            data?.analysis_phase ===
                "manual_analysis_completed"
        ) {
            manualAnalysisButton.textContent =
                "ANALYSIS";

            manualAnalysisActive = false;

            return;
        }

        if (
            data?.analysis_phase ===
                "manual_analysis_error" ||
            data?.analysis_phase ===
                "manual_analysis_paused"
        ) {
            manualAnalysisButton.textContent =
                "ANALYSIS";

            manualAnalysisActive = false;
        }
    }

    function updateResultsContext(data) {
        if (!resultsContext) {
            return;
        }

        const symbol =
            String(
                data?.symbol || ""
            ).toUpperCase();

        const timeframe =
            String(
                data?.timeframe || ""
            ).toUpperCase();

        let resultType = "TESTING";

        if (data?.results_source === "manual") {
            resultType = "MANUAL";
        } else if (
            data?.results_source === "testing"
        ) {
            resultType = "TESTING";
        }

        resultsContext.textContent =
            `${symbol}, ${timeframe}, ${resultType}`;
    }

    function render(data) {
        const isLoading =
            data?.startup_stage === "Loading market data" ||
            data?.startup_stage === "Starting" ||
            data?.loading === true;

        updateLoading(data);

        if (isLoading) {
            document.getElementById("status").textContent =
                data?.startup_message || "Loading market data...";
            return;
        }

        syncSelectorsFromBackend(data);

        updateResultsContext(data);

        if (!data.ready || !data.h1 || !data.line) {
            return;
        }

        const incomingSessionId = data.session_id ||
            `${data.symbol || ""}|${data.timeframe || ""}`;

        if (activeSessionId !== incomingSessionId) {
            activeSessionId = incomingSessionId;
            pendingSessionId = null;
            resetChartView();
        }

        lastData = data;

        const isManualResults =
            data?.results_source === "manual";

        const manualResults =
            data?.manual_analysis_results || {};

        const resultStartingBalance =
            isManualResults
                ? Number(
                    manualResults.starting_balance ??
                    100000
                )
                : Number(
                    data.starting_balance ??
                    100000
                );

        const resultFinalBalance =
            isManualResults
                ? Number(
                    manualResults.final_balance ??
                    resultStartingBalance
                )
                : Number(
                    data.final_balance ??
                    0
                );

        const resultReturn =
            isManualResults
                ? Number(
                    manualResults.total_return ??
                    0
                )
                : Number(
                    data.total_return ??
                    0
                );

        const resultTrades =
            isManualResults &&
            Array.isArray(manualResults.trades)
                ? manualResults.trades
                : (data.trades || []);

        document.getElementById("startingBalance").textContent =
            `${resultStartingBalance.toLocaleString("sk-SK")} €`;

        document.getElementById("finalBalance").textContent =
            `${resultFinalBalance.toLocaleString("sk-SK")} €`;

        document.getElementById("totalReturn").textContent =
            `${resultReturn >= 0 ? "+" : ""}${resultReturn.toFixed(2)}%`;

        document.getElementById("blockedTrades").textContent =
            isManualResults
                ? 0
                : Number(data.blocked_trades ?? 0);

        if (!chartDataLoaded) {
            candleSeries.setData(
                data.h1.map(candle => ({
                    time: candle.time,
                    open: candle.open,
                    high: candle.high,
                    low: candle.low,
                    close: candle.close
                }))
            );

            chartDataLoaded = true;
            chart.timeScale().fitContent();
            
            if (
                learningMode &&
                learningDetailView.style.display !== "none"
            ) {
                requestAnimationFrame(() => {
                    centerLearningChartOnAnchor();
                    renderLearningAnchor();
                });
            }
        }

        updateResults(resultTrades);

        document.getElementById("status").textContent =
            getStatusText(data);

        requestAnimationFrame(renderOverlay);
    }

    chart.timeScale().subscribeVisibleTimeRangeChange((range) => {
        requestAnimationFrame(() => {
            renderOverlay();

            if (
                learningMode &&
                learningDetailView.style.display !== "none" &&
                selectedLearningTradeId !== null
            ) {
                renderLearningGeometry();
            }
        });

        if (!range || !socket || socket.readyState !== WebSocket.OPEN) {
            return;
        }

        clearTimeout(viewportTimer);

        viewportTimer = setTimeout(() => {
            socket.send(JSON.stringify({
                action: "viewport",
                from: Math.floor(range.from),
                to: Math.floor(range.to)
            }));
        }, 1000);
    });

    window.addEventListener("resize", () => {
        chart.applyOptions({
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight
        });
        requestAnimationFrame(renderOverlay);
    });

    function renderLearningTradeList() {

        learningTradeList.innerHTML = "";

        const sortedTrades = [...learningTrades];

        if (!sortedTrades.length) {

            const empty = document.createElement("div");

            empty.className = "learning-trade-empty";
            empty.textContent = "NO LEARNING TRADES";

            learningTradeList.appendChild(empty);

            learningTradeCount.textContent = "0";

            return;
        }

        learningTradeCount.textContent =
            String(sortedTrades.length);

        sortedTrades.forEach((trade, index) => {

            const button =
                document.createElement("button");

            button.type = "button";
            button.className = "learning-trade-item";

            button.textContent =
                `${index + 1}. ${trade.direction || "LONG"}`;

            button.addEventListener("click", () => {

                openLearningTradeDetail(
                    trade.id
                );

            });

            learningTradeList.appendChild(button);

        });
    }


    function openLearningTradeDetail(tradeId) {

        const trade =
            learningTrades.find(
                item => item.id === tradeId
            );

        if (!trade) {
            return;
        }

        drawTradeSetupButton?.classList.remove("selected");

        selectedLearningTradeId =
            trade.id;

        const tradeIndex =
            learningTrades.findIndex(
                item => item.id === trade.id
            );

        learningDetailTradeNumber.textContent =
            String(tradeIndex + 1);

        learningSetupTimeframe.value =
            trade.setup_timeframe || "H1";

        learningEntryTimeframe.value =
            trade.entry_timeframe || "M15";

        learningAnchorTime.value =
            trade.anchor?.time
                ? formatAnchorTime(trade.anchor.time)
                : "";

        learningAnchorPrice.value =
            trade.anchor?.price ?? "";

        const rrMode =
            trade.rr_mode || "OFF";

        learningRRModeButtons.forEach(button => {
            button.classList.toggle(
                "active",
                button.dataset.rrMode === rrMode
            );
        });

        ensureLearningGeometry(trade);

        updateGeometrySummary();

        learningListView.style.display =
            "none";

        learningDetailView.style.display =
            "block";

        sidebar.classList.remove("collapsed");

        pendingAnchorCenter =
            !!trade.anchor?.time;

        requestAnimationFrame(() => {
            renderLearningAnchor();
            renderLearningGeometry();
            centerLearningChartOnAnchor();
        });

        sidebarToggle.querySelector("img").src = "images/left.png";
    }

    function closeLearningTradeDetail() {

        cancelPricePicker();

        if (selectedLearningTradeIsNew) {

            learningTrades =
                learningTrades.filter(
                    trade =>
                        trade.id !==
                        selectedLearningTradeId
                );
        }

        selectedLearningTradeId = null;
        syncGeometryButtonStates();

        selectedLearningTradeIsNew = false;

        learningDetailView.style.display =
            "none";

        learningListView.style.display =
            "block";

        renderLearningTradeList();
    }

    function createLearningTrade() {

        const trade = {

            id: nextLearningTradeId++,
            
            direction: null,

            setup_timeframe:
                timeframeSelect.value || "H1",

            entry_timeframe: "M15",

            anchor: {
                time: null,
                price: null
            },

            rr_mode: "OFF",

            zones: [],

            trade_geometry: {

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

            },

            breakout: null,
            entry_candle: "",

            entry: null,
            sl: null,
            tp: null
        };

        learningTrades.push(trade);

        selectedLearningTradeIsNew = true;

        openLearningTradeDetail(trade.id);
    }


    function saveLearningTrade() {

        if (selectedLearningTradeId === null) {
            return;
        }

        const trade =
            learningTrades.find(
                item =>
                    item.id ===
                    selectedLearningTradeId
            );

        if (!trade) {
            return;
        }

        trade.setup_timeframe =
            learningSetupTimeframe.value;

        trade.entry_timeframe =
            learningEntryTimeframe.value;

        ensureLearningGeometry(trade);

        trade.anchor = {
            time: trade.anchor?.time ?? null,
            price: trade.anchor?.price ?? null
        };

        trade.anchor = {
            time:
                trade.anchor?.time ?? null,

            price:
                trade.anchor?.price ?? null
        };

        if (
            !socket ||
            socket.readyState !== WebSocket.OPEN
        ) {
            console.error("Backend is not connected.");
            return;
        }

        ensureLearningGeometry(trade);

        if (trade.trade_geometry) {

            trade.breakout =
                trade.trade_geometry.breakout?.price
                ?? null;

            trade.entry =
                trade.trade_geometry.entry?.price
                ?? null;

            trade.sl =
                trade.trade_geometry.sl?.price
                ?? null;

            trade.tp =
                trade.trade_geometry.tp?.price
                ?? null;
        }

        socket.send(JSON.stringify({
            action: "learning_save",
            setup: trade
        }));

        selectedLearningTradeIsNew = false;

        closeLearningTradeDetail();
    }

    function formatAnchorTime(timestamp) {

        if (
            timestamp == null ||
            !Number.isFinite(Number(timestamp))
        ) {
            return "";
        }

        const date =
            new Date(Number(timestamp) * 1000);

        return date.toISOString()
            .replace("T", " ")
            .replace(".000Z", " UTC");
    }

    function updateLearningHeader() {
        learningMarket.textContent = marketSelect.value || "";
        learningTimeframe.textContent = timeframeSelect.value || "";
    }

    function cancelPricePicker() {

        activePriceTarget = null;

        document
            .querySelectorAll(".learning-price-picker.active")
            .forEach(button => {
                button.classList.remove("active");
            });

        chartContainer.classList.remove(
            "price-picker-active"
        );
    }


    function activatePricePicker(targetId, button) {

        cancelPricePicker();

        activePriceTarget = targetId;

        button.classList.add("active");

        chartContainer.classList.add(
            "price-picker-active"
        );
    }
  
    function pickPriceFromChart(event) {

        if (!activePriceTarget) {
            return false;
        }

        let y;

        if (isMobileTouchDevice()) {

            // Mobile: allow price selection only inside
            // the main candlestick plot, not on the axes.
            if (
                !isInsideMainPlot(
                    event.clientX,
                    event.clientY
                )
            ) {
                return false;
            }

            const plotRect = getMainPlotRect();

            y = event.clientY - plotRect.top;

            if (
                y < 0 ||
                y >= plotRect.bottom - plotRect.top
            ) {
                return false;
            }

        } else {

            // Preserve the existing desktop behavior.
            const rect =
                chartContainer.getBoundingClientRect();

            y = event.clientY - rect.top;

            if (
                y < 0 ||
                y > rect.height
            ) {
                return false;
            }
        }

        const price =
            candleSeries.coordinateToPrice(y);

        if (
            price == null ||
            !Number.isFinite(Number(price))
        ) {
            return false;
        }

        const input =
            document.getElementById(
                activePriceTarget
            );

        if (!input) {
            cancelPricePicker();
            return false;
        }

        input.value =
            Number(price).toFixed(5);

        input.dispatchEvent(
            new Event("input", {
                bubbles: true
            })
        );

        cancelPricePicker();

        return true;
    }
   
    function syncAnchorButtonState() {
        const trade = learningTrades.find(
            item => item.id === selectedLearningTradeId
        );

        const hasAnchor =
            trade?.anchor?.time != null &&
            trade?.anchor?.price != null;

        learningAnchorPicker.classList.toggle(
            "has-anchor",
            hasAnchor
        );
    }

    function cancelAnchorPicker() {
        activeAnchorPicker = false;

        learningAnchorPicker.classList.remove("active");

        syncAnchorButtonState();

        chartContainer.classList.remove(
            "price-picker-active"
        );
    }

    function activateAnchorPicker() {
        cancelPricePicker();
        cancelAnchorPicker();

        activeAnchorPicker = true;

        learningAnchorPicker.classList.add("active");
        learningAnchorPicker.classList.remove("has-anchor");

        chartContainer.classList.add(
            "price-picker-active"
        );
    }
   
    function pickAnchorFromChart(event) {

        if (!activeAnchorPicker) {
            return false;
        }

        let x;
        let y;

        if (isMobileTouchDevice()) {

            // On mobile, accept anchor selection only inside
            // the main candlestick plot, never on the axes.
            if (
                !isInsideMainPlot(
                    event.clientX,
                    event.clientY
                )
            ) {
                return false;
            }

            const plotRect = getMainPlotRect();

            x = event.clientX - plotRect.left;
            y = event.clientY - plotRect.top;

            if (
                x < 0 ||
                x >= plotRect.right - plotRect.left ||
                y < 0 ||
                y >= plotRect.bottom - plotRect.top
            ) {
                return false;
            }

        } else {

            // Preserve the existing desktop behavior.
            const rect =
                chartContainer.getBoundingClientRect();

            x = event.clientX - rect.left;
            y = event.clientY - rect.top;

            if (
                x < 0 ||
                x > rect.width ||
                y < 0 ||
                y > rect.height
            ) {
                return false;
            }
        }

        const rawTime =
            chart.timeScale().coordinateToTime(x);

        const price =
            candleSeries.coordinateToPrice(y);

        if (
            rawTime == null ||
            price == null
        ) {
            return false;
        }

        const timestamp = Number(rawTime);
        const numericPrice = Number(price);

        if (
            !Number.isFinite(timestamp) ||
            !Number.isFinite(numericPrice)
        ) {
            return false;
        }

        const trade =
            learningTrades.find(
                item =>
                    item.id === selectedLearningTradeId
            );

        if (!trade) {
            return false;
        }

        trade.anchor = {
            time: Math.floor(timestamp),
            price: numericPrice
        };

        learningAnchorTime.value =
            formatAnchorTime(trade.anchor.time);

        learningAnchorPrice.value =
            numericPrice.toFixed(5);

        pendingAnchorCenter = false;

        cancelAnchorPicker();

        renderLearningAnchor();

        return true;
    }

    function renderLearningAnchor() {

        overlay
            .querySelectorAll(
                ".learning-anchor-point, .learning-anchor-label"
            )
            .forEach(el => el.remove());

        if (
            selectedLearningTradeId === null
        ) {
            return;
        }

        const trade =
            learningTrades.find(
                item =>
                    item.id ===
                    selectedLearningTradeId
            );

        if (
            !trade ||
            !trade.anchor ||
            trade.anchor.time == null ||
            trade.anchor.price == null
        ) {
            return;
        }

       const x =
            timeCoord(trade.anchor.time);

        const y =
            priceCoord(trade.anchor.price);

        if (
            x == null ||
            y == null
        ) {
            return;
        }

        let markerX = x;
        let markerY = y;

        if (isMobileTouchDevice()) {
            const containerRect =
                chartContainer.getBoundingClientRect();

            const plotRect = getMainPlotRect();

            markerX += plotRect.left - containerRect.left;
            markerY += plotRect.top - containerRect.top;
        }

        const point =
            document.createElement("div");

        point.className =
            "learning-anchor-point";

        point.style.left = `${markerX}px`;
        point.style.top = `${markerY}px`;

        overlay.appendChild(point);

        const label =
            document.createElement("div");

        label.className =
            "learning-anchor-label";

        label.textContent =
            `ANCHOR ${fmtPrice(trade.anchor.price)}`;

        label.style.left =
            `${markerX + 10}px`;

        label.style.top =
            `${markerY - 22}px`;

        overlay.appendChild(label);
    }

    /* =========================================================
    LEARNING GEOMETRY STATE
    ========================================================= */

    let activeGeometryTool = null;
    let geometryDrag = null;

    let activeZoneType = null;
    let nextLearningZoneId = 1;

    /* =========================================================
    GEOMETRY HELPERS
    ========================================================= */

    function getSelectedLearningTrade() {
        if (selectedLearningTradeId === null) {
            return null;
        }

        return learningTrades.find(
            trade => trade.id === selectedLearningTradeId
        ) || null;
    }


    function ensureLearningGeometry(trade) {
        if (!trade) {
            return null;
        }

        if (!Array.isArray(trade.zones)) {
            trade.zones = [];
        }

        if (!trade.trade_geometry) {
            trade.trade_geometry = {
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

        return trade;
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

    function renumberLearningZones(trade) {
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
            }

            if (zone.type === "RESISTANCE") {
                resistanceNumber++;
                zone.number = resistanceNumber;
            }
        }
    }

    function getZoneLabel(zone) {
        return `${zone.type} ${zone.number}`;
    }

    function renderLearningZoneList() {
        learningZoneList.innerHTML = "";

        const trade =
            getSelectedLearningTrade();

        if (!trade) {
            return;
        }

        ensureLearningGeometry(trade);
        renumberLearningZones(trade);

        for (const zone of trade.zones) {
            const item =
                document.createElement("div");

            item.className =
                "learning-zone-item";

            item.dataset.zoneId =
                String(zone.id);

            const header =
                document.createElement("div");

            header.className =
                "learning-zone-header";

            const mainButton =
                document.createElement("button");

            mainButton.type = "button";
            mainButton.className =
                "learning-zone-main";

            mainButton.textContent =
                getZoneLabel(zone);

            mainButton.style.borderLeft =
                `3px solid ${zone.color}`;

            const expandButton =
                document.createElement("button");

            expandButton.type = "button";
            expandButton.className =
                "learning-zone-expand";

            expandButton.textContent = "▼";

            const deleteButton =
                document.createElement("button");

            deleteButton.type = "button";
            deleteButton.className =
                "learning-zone-delete";

            deleteButton.textContent = "×";
            deleteButton.title =
                `Delete ${getZoneLabel(zone)}`;

            const body =
                document.createElement("div");

            body.className =
                "learning-zone-body";

            const geometry =
                normalizeGeometryRect(
                    zone.geometry
                );

            body.textContent =
                geometry
                    ? `${formatGeometryTime(geometry.start_time)} → ${formatGeometryTime(geometry.end_time)} | ${fmtPrice(geometry.low)} - ${fmtPrice(geometry.high)}`
                    : "Not set";

            mainButton.addEventListener(
                "click",
                () => {
                    setGeometryTool("zone", zone.type);
                }
            );

            expandButton.addEventListener(
                "click",
                event => {
                    event.preventDefault();
                    event.stopPropagation();

                    const isOpen =
                        item.classList.toggle("open");

                    expandButton.textContent =
                        isOpen ? "▲" : "▼";
                }
            );

            deleteButton.addEventListener(
                "click",
                event => {
                    event.preventDefault();
                    event.stopPropagation();

                    trade.zones =
                        trade.zones.filter(
                            current =>
                                current.id !== zone.id
                        );

                    renumberLearningZones(trade);

                    setGeometryTool(null);

                    renderLearningZoneList();
                    renderLearningGeometry();
                    updateGeometrySummary();
                }
            );

            header.appendChild(mainButton);
            header.appendChild(expandButton);
            header.appendChild(deleteButton);

            item.appendChild(header);
            item.appendChild(body);

            learningZoneList.appendChild(item);
        }
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


    function geometryTimeToX(time) {
        if (time == null) {
            return null;
        }

        const timestamp = Number(time);

        if (!Number.isFinite(timestamp)) {
            return null;
        }

        const x = chart.timeScale().timeToCoordinate(timestamp);

        return x == null
            ? null
            : Number(x);
    }


    function geometryPriceToY(price) {
        if (price == null) {
            return null;
        }

        return candleSeries.priceToCoordinate(
            Number(price)
        );
    }


    function geometryXToTime(x) {
        const rawTime =
            chart.timeScale().coordinateToTime(x);

        if (rawTime == null) {
            return null;
        }

        const time = Number(rawTime);

        return Number.isFinite(time)
            ? Math.floor(time)
            : null;
    }

    
    function geometryYToPrice(y) {
        const containerRect =
            chartContainer.getBoundingClientRect();

        const plotRect = getMainPlotRect();

        const plotY =
            Number(y) - (plotRect.top - containerRect.top);

        const price =
            candleSeries.coordinateToPrice(plotY);

        if (price == null) {
            return null;
        }

        const value = Number(price);

        return Number.isFinite(value)
            ? value
            : null;
    }

    function clamp(value, min, max) {
        return Math.max(min, Math.min(max, value));
    }


    function interpolate(start, end, ratio) {
        return start + (end - start) * ratio;
    }


    function geometryTimeRatio(time, startTime, endTime) {
        if (
            time == null ||
            startTime == null ||
            endTime == null
        ) {
            return 0.5;
        }

        const span =
            Number(endTime) -
            Number(startTime);

        if (!Number.isFinite(span) || span === 0) {
            return 0.5;
        }

        return clamp(
            (Number(time) - Number(startTime)) / span,
            0,
            1
        );
    }


    /* =========================================================
    GEOMETRY SUMMARY
    ========================================================= */

    function formatGeometryTime(time) {
        if (time == null) {
            return "-";
        }

        return formatTime(time);
    }


    function updateGeometrySummary() {
        const trade = getSelectedLearningTrade();

        if (!trade) {
            return;
        }

        ensureLearningGeometry(trade);
        renumberLearningZones(trade);
        renderLearningZoneList();

        const setup =
            trade.trade_geometry;

        const tradeSetupSummary =
            document.getElementById("tradeSetupSummary");

        const breakoutValue =
            document.getElementById("learningBreakoutValue");

        const entryValue =
            document.getElementById("learningEntryValue");

        const slValue =
            document.getElementById("learningSLValue");

        const tpValue =
            document.getElementById("learningTPValue");

        if (tradeSetupSummary) {
            if (
                setup &&
                setup.start_time != null &&
                setup.end_time != null
            ) {
                tradeSetupSummary.textContent =
                    `${formatGeometryTime(setup.start_time)} → ${formatGeometryTime(setup.end_time)}`;
            } else {
                tradeSetupSummary.textContent = "Not set";
            }
        }


        if (breakoutValue) {
            breakoutValue.textContent =
                setup?.breakout?.price != null
                    ? `${fmtPrice(setup.breakout.price)} | ${formatGeometryTime(setup.breakout.time)}`
                    : "-";
        }


        if (entryValue) {
            entryValue.textContent =
                setup?.entry?.price != null
                    ? `${fmtPrice(setup.entry.price)} | ${formatGeometryTime(setup.entry.time)}`
                    : "-";
        }


        if (slValue) {
            slValue.textContent =
                setup?.sl?.price != null
                    ? `${fmtPrice(setup.sl.price)} | ${formatGeometryTime(setup.sl.time)}`
                    : "-";
        }


        if (tpValue) {
            tpValue.textContent =
                setup?.tp?.price != null
                    ? `${fmtPrice(setup.tp.price)} | ${formatGeometryTime(setup.tp.time)}`
                    : "-";
        }
    }


    /* =========================================================
    DRAW RECTANGLE
    ========================================================= */

    function drawGeometryRect(
        rect,
        className,
        label,
        objectType,
        zoneId = null,
        zoneColor = null
    ){
        const normalized =
            normalizeGeometryRect(rect);

        if (!normalized) {
            return;
        }

        const x1 =
            geometryTimeToX(
                normalized.start_time
            );

        const x2 =
            geometryTimeToX(
                normalized.end_time
            );

        const y1 =
            geometryPriceToY(
                normalized.high
            );

        const y2 =
            geometryPriceToY(
                normalized.low
            );

        if (
            x1 == null ||
            x2 == null ||
            y1 == null ||
            y2 == null
        ) {
            return;
        }

        const left =
            Math.min(x1, x2);

        const right =
            Math.max(x1, x2);

        const top =
            Math.min(y1, y2);

        const bottom =
            Math.max(y1, y2);

        const width =
            Math.max(
                8,
                right - left
            );

        const height =
            Math.max(
                8,
                bottom - top
            );

        const rectangle =
            document.createElement("div");

        rectangle.className =
            `learning-geometry-rect ${className}`;

        const geometryGroup =
            objectType === "zone"
                ? `zone:${zoneId}`
                : objectType;

        rectangle.dataset.geometryGroup =
            geometryGroup;

        rectangle.dataset.geometryObject =
            objectType;
        
        if (zoneId !== null) {
            rectangle.dataset.geometryZoneId =
                String(zoneId);
        }

        if (zoneColor) {
            rectangle.style.setProperty(
                "--zone-color",
                zoneColor
            );

            rectangle.style.setProperty(
                "--zone-bg",
                `${zoneColor}24`
            );

            rectangle.style.setProperty(
                "--zone-glow",
                `${zoneColor}55`
            );
        }

        rectangle.style.left =
            `${left}px`;

        rectangle.style.top =
            `${top}px`;

        rectangle.style.width =
            `${width}px`;

        rectangle.style.height =
            `${height}px`;

        rectangle.title =
            `Drag ${label}`;

        rectangle.addEventListener(
            "mousedown",
            beginGeometryObjectInteraction
        );

        overlay.appendChild(
            rectangle
        );

        const labelElement =
            document.createElement("div");

        labelElement.className =
            "learning-setup-label";

        labelElement.dataset.geometryGroup =
            geometryGroup;

        labelElement.textContent =
            label;

        labelElement.style.left =
            `${left + 6}px`;

        labelElement.style.top =
            `${top + 6}px`;

        labelElement.style.pointerEvents =
            "none";

        overlay.appendChild(
            labelElement
        );

        createFocusResizeHandles(
            normalized,
            objectType,
            left,
            top,
            width,
            height,
            geometryGroup,
            zoneId
        );
    }

    function createFocusResizeHandles(
        rect,
        objectType,
        left,
        top,
        width,
        height,
        geometryGroup = null,
        zoneId = null
    ) {
        const handles = [
            {
                edge: "left",
                left: left - 4,
                top: top,
                width: 8,
                height: height
            },
            {
                edge: "right",
                left: left + width - 4,
                top: top,
                width: 8,
                height: height
            },
            {
                edge: "top",
                left: left,
                top: top - 4,
                width: width,
                height: 8
            },
            {
                edge: "bottom",
                left: left,
                top: top + height - 4,
                width: width,
                height: 8
            }
        ];

        handles.forEach(handleInfo => {
            const handle =
                document.createElement("div");

            handle.className =
                `learning-geometry-resize-handle ${handleInfo.edge}`;

            handle.dataset.geometryObject =
                objectType;

            if (zoneId !== null) {
                handle.dataset.geometryZoneId =
                    String(zoneId);
            }
                
            if (geometryGroup) {
                handle.dataset.geometryGroup =
                    geometryGroup;
            }

            handle.dataset.geometryEdge =
                handleInfo.edge;

            handle.style.left =
                `${handleInfo.left}px`;

            handle.style.top =
                `${handleInfo.top}px`;

            handle.style.width =
                `${handleInfo.width}px`;

            handle.style.height =
                `${handleInfo.height}px`;

            handle.addEventListener(
                "mousedown",
                beginGeometryObjectInteraction
            );

            overlay.appendChild(handle);
        });
    }


    /* =========================================================
    DRAW TRADE SETUP
    ========================================================= */

    function drawTradeSetupGeometry(geometry) {
        if (!geometry) {
            return;
        }

        if (
            geometry.start_time == null ||
            geometry.end_time == null
        ) {
            return;
        }

        const x1 =
            geometryTimeToX(
                geometry.start_time
            );

        const x2 =
            geometryTimeToX(
                geometry.end_time
            );

        if (
            x1 == null ||
            x2 == null
        ) {
            return;
        }

        const prices = [
            geometry.breakout?.price,
            geometry.entry?.price,
            geometry.sl?.price,
            geometry.tp?.price
        ]
            .filter(
                price =>
                    price != null &&
                    Number.isFinite(
                        Number(price)
                    )
            )
            .map(Number);

        if (!prices.length) {
            return;
        }

        const high =
            Math.max(...prices);

        const low =
            Math.min(...prices);

        const y1 =
            geometryPriceToY(high);

        const y2 =
            geometryPriceToY(low);

        if (
            y1 == null ||
            y2 == null
        ) {
            return;
        }

        const left =
            Math.min(x1, x2);

        const right =
            Math.max(x1, x2);

        const top =
            Math.min(y1, y2);

        const bottom =
            Math.max(y1, y2);

        const width =
            Math.max(
                12,
                right - left
            );

        const height =
            Math.max(
                20,
                bottom - top
            );

        const rectangle =
            document.createElement("div");

        rectangle.className =
            "learning-geometry-rect learning-trade-setup-rect";

        rectangle.dataset.geometryObject =
            "trade";

        rectangle.dataset.geometryGroup =
            "trade";

        rectangle.style.left =
            `${left}px`;

        rectangle.style.top =
            `${top}px`;

        rectangle.style.width =
            `${width}px`;

        rectangle.style.height =
            `${height}px`;

        rectangle.title =
            "Drag whole TRADE SETUP";

        rectangle.addEventListener(
            "mousedown",
            beginGeometryObjectInteraction
        );

        overlay.appendChild(
            rectangle
        );

        const entryY = geometry.entry?.price != null
            ? geometryPriceToY(geometry.entry.price)
            : null;

        function addSetupArea(targetPrice, className) {
            if (entryY == null || targetPrice == null) {
                return;
            }

            const targetY = geometryPriceToY(targetPrice);

            if (targetY == null) {
                return;
            }

            const area = document.createElement("div");
            area.className = `learning-setup-area ${className}`;
            area.dataset.geometryGroup = "trade";

            area.style.left = `${left}px`;
            area.style.top = `${Math.min(entryY, targetY)}px`;
            area.style.width = `${width}px`;
            area.style.height = `${Math.max(1, Math.abs(targetY - entryY))}px`;

            overlay.appendChild(area);
        }

        addSetupArea(
            geometry.tp?.price,
            "learning-setup-profit-area"
        );

        addSetupArea(
            geometry.sl?.price,
            "learning-setup-risk-area"
        );

        createTradeSetupEdge(
            "left",
            left,
            top,
            height,
            left
        );

        createTradeSetupEdge(
            "right",
            right - 6,
            top,
            height,
            right
        );

        const levels = [
            {
                key: "breakout",
                label: "BREAKOUT",
                className:
                    "learning-setup-breakout"
            },
            {
                key: "entry",
                label: "ENTRY",
                className:
                    "learning-setup-entry"
            },
            {
                key: "sl",
                label: "SL",
                className:
                    "learning-setup-sl"
            },
            {
                key: "tp",
                label: "TP",
                className:
                    "learning-setup-tp"
            }
        ];

        levels.forEach(level => {
            const point =
                geometry[level.key];

            if (
                !point ||
                point.price == null
            ) {
                return;
            }

            const y =
                geometryPriceToY(
                    point.price
                );

            if (y == null) {
                return;
            }

            const line =
                document.createElement("div");

            line.className =
                `learning-setup-level ${level.className}`;

            line.dataset.geometryLevel =
                level.key;

            line.dataset.geometryGroup =
                "trade";

            line.style.left =
                `${left}px`;

            line.style.width =
                `${width}px`;

            line.style.top =
                `${y - 2}px`;

            line.title =
                `Drag ${level.label}`;

            line.addEventListener(
                "mousedown",
                beginGeometryObjectInteraction
            );

            overlay.appendChild(
                line
            );

            const labelElement =
                document.createElement("div");

            labelElement.className =
                "learning-setup-label";

            labelElement.dataset.geometryGroup =
                "trade";

            labelElement.textContent =
                `${level.label} ${fmtPrice(point.price)} | ${formatGeometryTime(point.time)}`;

            labelElement.style.left =
                `${left + 6}px`;

            labelElement.style.top =
                `${y - 18}px`;

            labelElement.style.pointerEvents =
                "none";

            overlay.appendChild(
                labelElement
            );
        });
    }

    function createTradeSetupEdge(
        edge,
        left,
        top,
        height,
        visualLeft
    ) {
        const handle =
            document.createElement("div");

        handle.className =
            `learning-trade-setup-edge ${edge}`;

        handle.dataset.geometryObject =
            "trade";

        handle.dataset.geometryGroup =
            "trade";

        handle.dataset.geometryEdge =
            edge;

        handle.style.left =
            `${visualLeft - 3}px`;

        handle.style.top =
            `${top}px`;

        handle.style.height =
            `${height}px`;

        handle.addEventListener(
            "mousedown",
            beginGeometryObjectInteraction
        );

        overlay.appendChild(
            handle
        );
    }


    /* =========================================================
    MAIN LEARNING GEOMETRY RENDER
    ========================================================= */

    function syncGeometryButtonStates() {
        const trade = getSelectedLearningTrade();

        const focus1Button =
            document.getElementById("drawFocus1Button");

        const focus2Button =
            document.getElementById("drawFocus2Button");

        const tradeSetupButton =
            document.getElementById("drawTradeSetupButton");

        const hasFocus1 =
            !!trade?.focus_1_geometry;

        const hasFocus2 =
            !!trade?.focus_2_geometry;

        const hasTradeSetup =
            !!trade?.trade_geometry &&
            trade.trade_geometry.start_time != null &&
            trade.trade_geometry.end_time != null;

        focus1Button?.classList.toggle(
            "active",
            hasFocus1
        );

        focus2Button?.classList.toggle(
            "active",
            hasFocus2
        );

        tradeSetupButton?.classList.toggle(
            "active",
            hasTradeSetup
        );
    }

    function renderLearningGeometry() {

        syncGeometryButtonStates();

        overlay
            .querySelectorAll(
                ".learning-geometry-rect, " +
                ".learning-setup-area, " +
                ".learning-setup-level, " +
                ".learning-setup-label, " +
                ".learning-geometry-resize-handle, " +
                ".learning-trade-setup-edge"
            )
            .forEach(
                element => element.remove()
            );


        if (
            !learningMode ||
            learningDetailView.style.display === "none" ||
            selectedLearningTradeId === null
        ) {
            return;
        }


        const trade =
            getSelectedLearningTrade();

        if (!trade) {
            return;
        }

        ensureLearningGeometry(trade);


        renumberLearningZones(trade);

        trade.zones.forEach(zone => {
            drawGeometryRect(
                zone.geometry,
                "learning-zone-rect",
                getZoneLabel(zone),
                "zone",
                zone.id,
                zone.color
            );
        });

        drawTradeSetupGeometry(
            trade.trade_geometry
        );

        updateGeometrySummary();
    }

    /* =========================================================
    GEOMETRY TOOL SELECTION
    ========================================================= */

    function setGeometryTool(
        tool,
        zoneType = null
    ) {
        activeGeometryTool =
            tool;

        activeZoneType =
            zoneType;

        chartContainer.classList.remove(
            "learning-geometry-editing"
        );

        if (!activeGeometryTool) {
            return;
        }

        chartContainer.classList.add(
            "learning-geometry-editing"
        );
    }

    /* =========================================================
    START DRAW
    ========================================================= */
    
    function getChartPaneLeftOffset() {
        const chartWidth =
            chartContainer.clientWidth;

        const timeScaleWidth =
            chart.timeScale().width();

        return Math.max(
            0,
            chartWidth - timeScaleWidth
        );
    }
    
    function getMainPlotRect() {
        const containerRect =
            chartContainer.getBoundingClientRect();

        const plotWidth = Math.max(
            1,
            chart.timeScale().width()
        );

        const tolerance = Math.max(
            6,
            plotWidth * 0.02
        );

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

        const left =
            containerRect.left + getChartPaneLeftOffset();

        return {
            left,
            right: left + plotWidth,
            top: containerRect.top,
            bottom: containerRect.bottom - 28
        };
    }

    function isInsideMainPlot(clientX, clientY) {
        const rect = getMainPlotRect();

        return (
            clientX >= rect.left &&
            clientX < rect.right &&
            clientY >= rect.top &&
            clientY < rect.bottom
        );
    }



    function geometryTimeToX(time) {
        if (time == null) {
            return null;
        }

        const timestamp = Number(time);

        if (!Number.isFinite(timestamp)) {
            return null;
        }

        const localX =
            chart.timeScale().timeToCoordinate(
                timestamp
            );

        if (localX == null) {
            return null;
        }

        return Number(localX) +
            getChartPaneLeftOffset();
    }


    function geometryXToTime(x) {
        const localX =
            Number(x) -
            getChartPaneLeftOffset();

        const rawTime =
            chart.timeScale().coordinateToTime(
                localX
            );

        if (rawTime == null) {
            return null;
        }

        const time = Number(rawTime);

        return Number.isFinite(time)
            ? Math.floor(time)
            : null;
    }


    function snapXToCandleTime(x) {
        if (!lastData?.h1?.length) {
            return null;
        }

        let nearestTime = null;
        let nearestDistance = Infinity;

        for (const candle of lastData.h1) {
            const candleX =
                geometryTimeToX(
                    Number(candle.time)
                );

            if (candleX == null) {
                continue;
            }

            const distance =
                Math.abs(candleX - x);

            if (distance < nearestDistance) {
                nearestDistance = distance;
                nearestTime = Number(candle.time);
            }
        }

        return nearestTime;
    }

    function geometryPointerToChartPoint(event) {
        if (
            isMobileTouchDevice() &&
            !isInsideMainPlot(event.clientX, event.clientY)
        ) {
            return null;
        }

        const rect =
            chartContainer.getBoundingClientRect();

        const x =
            event.clientX - rect.left;

        const y =
            event.clientY - rect.top;

        if (
            x < 0 ||
            x > rect.width ||
            y < 0 ||
            y > rect.height
        ) {
            return null;
        }

        const time =
            geometryXToTime(x);

        const price =
            geometryYToPrice(y);

        if (
            time == null ||
            price == null
        ) {
            return null;
        }

        return {
            x,
            y,
            time,
            price
        };
    }

    function beginGeometryDraw(event) {
        if (!activeGeometryTool) {
            return false;
        }

        const point =
            geometryPointerToChartPoint(event);

        if (!point) {
            return false;
        }

        const trade =
            getSelectedLearningTrade();

        geometryDrag = {
            mode: "draw",
            tool: activeGeometryTool,
            zoneType: activeZoneType,
            zoneColor:
                activeGeometryTool === "zone" &&
                trade
                    ? getZoneColor(activeZoneType)
                    : null,
            start: point,
            current: point
        };

        return true;
    }

    function updateGeometryDraw(event) {

        if (
            !geometryDrag ||
            geometryDrag.mode !== "draw"
        ) {
            return false;
        }

        const point =
            geometryPointerToChartPoint(
                event
            );

        if (!point) {
            return false;
        }

        geometryDrag.current =
            point;

        renderGeometryPreview();

        return true;
    }

    function syncTradeDirectionFromSetup(trade) {
        if (!trade?.trade_geometry) {
            return;
        }

        const startPrice =
            Number(trade.trade_geometry.sl?.price);

        const endPrice =
            Number(trade.trade_geometry.tp?.price);

        if (
            !Number.isFinite(startPrice) ||
            !Number.isFinite(endPrice) ||
            startPrice === endPrice
        ) {
            trade.direction = null;
            return;
        }

        trade.direction =
            endPrice > startPrice
                ? "LONG"
                : "SHORT";
    }

    function finishGeometryDraw(event) {

        if (
            !geometryDrag ||
            geometryDrag.mode !== "draw"
        ) {
            return false;
        }

        const point =
            geometryPointerToChartPoint(
                event
            );

        if (point) {
            geometryDrag.current =
                point;
        }

        const start =
            geometryDrag.start;

        const end =
            geometryDrag.current;

        const trade =
            getSelectedLearningTrade();

        if (!trade) {
            geometryDrag = null;
            return true;
        }

        ensureLearningGeometry(trade);


        const startTime =
            Math.min(
                start.time,
                end.time
            );

        const endTime =
            Math.max(
                start.time,
                end.time
            );

        const low =
            Math.min(
                start.price,
                end.price
            );

        const high =
            Math.max(
                start.price,
                end.price
            );


        if (geometryDrag.tool === "zone") {
            if (
                geometryDrag.zoneType !== "SUPPORT" &&
                geometryDrag.zoneType !== "RESISTANCE"
            ) {
                geometryDrag = null;
                setGeometryTool(null);
                return true;
            }

            const zone = {
                id: nextLearningZoneId++,

                type:
                    geometryDrag.zoneType,

                number: 0,

                color: getZoneColor(geometryDrag.zoneType),

                geometry: {
                    start_time: startTime,
                    end_time: endTime,
                    low,
                    high
                }
            };

            trade.zones.push(zone);

            renumberLearningZones(trade);

            renderLearningZoneList();

        } else if (geometryDrag.tool === "trade") {
            const startPrice =
                Number(start.price);

            const endPrice =
                Number(end.price);

            const lowPrice =
                Math.min(
                    startPrice,
                    endPrice
                );

            const highPrice =
                Math.max(
                    startPrice,
                    endPrice
                );

            const range =
                highPrice -
                lowPrice;

            if (
                !Number.isFinite(range) ||
                range <= 0
            ) {
                geometryDrag = null;
                setGeometryTool(null);
                return true;
            }

            const centerPrice =
                startPrice;

            const directionSign =
                endPrice > startPrice
                    ? 1
                    : -1;

            const slPrice =
                centerPrice;

            const breakoutPrice =
                centerPrice +
                directionSign *
                range *
                0.33;

            const entryPrice =
                centerPrice +
                directionSign *
                range *
                0.66;

            const tpPrice =
                endPrice;

            const leftTime =
                Math.min(
                    start.time,
                    end.time
                );

            const rightTime =
                Math.max(
                    start.time,
                    end.time
                );

            const timeRange =
                rightTime -
                leftTime;

            trade.trade_geometry = {
                start_time:
                    leftTime,

                end_time:
                    rightTime,

                breakout: {
                    time:
                        leftTime +
                        timeRange * 0.33,

                    price:
                        breakoutPrice
                },

                entry: {
                    time:
                        leftTime +
                        timeRange * 0.66,

                    price:
                        entryPrice
                },

                sl: {
                    time:
                        leftTime,

                    price:
                        slPrice
                },

                tp: {
                    time:
                        rightTime,

                    price:
                        tpPrice
                }
            };

            syncTradeDirectionFromSetup(trade);

            trade.breakout =
                trade.trade_geometry.breakout.price;

            trade.entry =
                trade.trade_geometry.entry.price;

            trade.sl =
                trade.trade_geometry.sl.price;

            trade.tp =
                trade.trade_geometry.tp.price;
        }


        geometryDrag = null;

        setGeometryTool(null);

        renderLearningGeometry();

        return true;
    }


    /* =========================================================
    TEMPORARY DRAW PREVIEW
    ========================================================= */

    function renderGeometryPreview() {

        if (!geometryDrag) {
            return;
        }

        overlay
            .querySelectorAll(
                ".learning-geometry-preview"
            )
            .forEach(
                element => element.remove()
            );


        const start =
            geometryDrag.start;

        const current =
            geometryDrag.current;

        const top =
            Math.min(
                start.y,
                current.y
            );

        const startX =
            geometryTimeToX(start.time);

        const currentX =
            geometryTimeToX(current.time);

        if (startX == null || currentX == null) {
            return;
        }

        const left =
            Math.min(
                startX,
                currentX
            );

        const width =
            Math.max(
                2,
                Math.abs(
                    currentX - startX
                )
            );

        const height =
            Math.max(
                2,
                Math.abs(
                    current.y - start.y
                )
            );


        const preview =
            document.createElement("div");

        preview.className =
            "learning-geometry-rect learning-geometry-preview";

        preview.style.left =
            `${left}px`;

        preview.style.top =
            `${top}px`;

        preview.style.width =
            `${width}px`;

        preview.style.height =
            `${height}px`;


        if (geometryDrag.tool === "zone") {
            const color =
                geometryDrag.zoneColor ||
                "#a456ff";

            preview.style.background =
                `${color}24`;

            preview.style.border =
                `1px dashed ${color}`;

            preview.style.setProperty(
                "--zone-color",
                color
            );

            preview.style.setProperty(
                "--zone-bg",
                `${color}24`
            );
        }    
        
        if (
            geometryDrag.tool === "trade"
        ) {
            preview.style.background =
                "rgba(42, 156, 255, 0.06)";

            preview.style.border =
                "1px dashed #2a9cff";
        }

        overlay.appendChild(preview);
    }

    function beginGeometryObjectInteraction(event) {
        if (
            event.button !== 0 ||
            geometryDrag
        ) {
            return;
        }

        const trade =
            getSelectedLearningTrade();

        if (!trade) {
            return;
        }

        const object =
            event.currentTarget;

        const objectType =
            object.dataset.geometryObject;

        const edge =
            object.dataset.geometryEdge ||
            null;

        const level =
            object.dataset.geometryLevel ||
            null;

        const point =
            geometryPointerToChartPoint(event);

        if (!point) {
            return;
        }

        const zoneId =
            object.dataset.geometryZoneId ||
            null;

        const geometryGroup =
            object.dataset.geometryGroup ||
            null;

        geometryDrag = {
            mode:
                level
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

            startClientX:
                event.clientX,

            startClientY:
                event.clientY,

            original: {
                zones:
                    (trade.zones || []).map(
                        zone => ({
                            ...zone,
                            geometry:
                                zone.geometry
                                    ? {
                                        ...zone.geometry
                                    }
                                    : null
                        })
                    ),

                trade:
                    trade.trade_geometry
                        ? JSON.parse(
                            JSON.stringify(
                                trade.trade_geometry
                            )
                        )
                        : null
            }
        };

        event.preventDefault();
        event.stopPropagation();

        document.body.style.userSelect =
            "none";

        if (
            geometryDrag.mode === "move"
        ) {
            document.body.style.cursor =
                "grabbing";

            overlay
                .querySelectorAll(
                    "[data-geometry-group]"
                )
                .forEach(element => {
                    if (
                        element.dataset.geometryGroup ===
                        geometryGroup
                    ) {
                        element.style.transition =
                            "none";
                    }
                });
        }
    }

    function applyTradeRRMode(
        trade,
        changedLevel,
        originalGeometry = null
    ) {
        const mode =
            trade?.rr_mode || "OFF";

        if (mode === "OFF") {
            return;
        }

        const geometry =
            trade.trade_geometry;

        if (!geometry?.entry?.price) {
            return;
        }

        const entry =
            Number(geometry.entry.price);

        const rr =
            mode === "2x"
                ? 2
                : mode === "3x"
                    ? 3
                    : null;

        if (
            rr == null ||
            !Number.isFinite(entry)
        ) {
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

            const originalEntry =
                Number(originalGeometry.entry.price);

            const originalSL =
                Number(originalGeometry.sl.price);

            const originalTP =
                Number(originalGeometry.tp.price);

            if (
                !Number.isFinite(originalEntry) ||
                !Number.isFinite(originalSL) ||
                !Number.isFinite(originalTP)
            ) {
                return;
            }

            const movement =
                entry - originalEntry;

            if (movement === 0) {
                return;
            }

            const movedTowardSL =
                movement * (originalSL - originalEntry) > 0;

            const movedTowardTP =
                movement * (originalTP - originalEntry) > 0;

            if (movedTowardSL) {
                const risk =
                    Math.abs(entry - originalSL);

                if (risk <= 0) {
                    return;
                }

                const direction =
                    entry > originalSL ? 1 : -1;

                geometry.sl.price = originalSL;

                geometry.tp.price =
                    entry + direction * risk * rr;

                return;
            }

            if (movedTowardTP) {
                const reward =
                    Math.abs(originalTP - entry);

                if (reward <= 0) {
                    return;
                }

                const direction =
                    originalTP > entry ? 1 : -1;

                geometry.tp.price = originalTP;

                geometry.sl.price =
                    entry - direction * (reward / rr);

                return;
            }
        }

        if (
            changedLevel === "sl" &&
            Number.isFinite(
                Number(geometry.sl?.price)
            )
        ) {
            const sl =
                Number(geometry.sl.price);

            const risk =
                Math.abs(entry - sl);

            if (risk <= 0) {
                return;
            }

            const direction =
                entry > sl
                    ? 1
                    : -1;

            geometry.tp.price =
                entry +
                direction *
                risk *
                rr;
        }

        if (
            changedLevel === "tp" &&
            Number.isFinite(
                Number(geometry.tp?.price)
            )
        ) {
            const tp =
                Number(geometry.tp.price);

            const reward =
                Math.abs(tp - entry);

            if (reward <= 0) {
                return;
            }

            const direction =
                tp > entry
                    ? 1
                    : -1;

            geometry.sl.price =
                entry -
                direction *
                (reward / rr);
        }
    }

    function updateGeometryObjectInteraction(
        event
    ) {
        if (!geometryDrag) {
            return false;
        }

        const trade =
            getSelectedLearningTrade();

        if (!trade) {
            return false;
        }

        /*
        * =========================================================
        * FREE DRAG MODE
        * =========================================================
        */
        if (
            geometryDrag.mode === "move"
        ) {
            const dx =
                event.clientX -
                geometryDrag.startClientX;

            const dy =
                event.clientY -
                geometryDrag.startClientY;

            const group =
                geometryDrag.geometryGroup;

            if (!group) {
                return false;
            }

            overlay
                .querySelectorAll(
                    "[data-geometry-group]"
                )
                .forEach(element => {
                    if (
                        element.dataset.geometryGroup ===
                        group
                    ) {
                        element.style.translate =
                            `${dx}px ${dy}px`;
                    }
                });

            event.preventDefault();
            event.stopPropagation();

            return true;
        }

        /*
        * =========================================================
        * EXISTING RESIZE / LEVEL LOGIC
        * =========================================================
        */

        const current =
            geometryPointerToChartPoint(
                event
            );

        if (!current) {
            return false;
        }

        geometryDrag.current =
            current;

        const original =
            geometryDrag.original;

        if (
            geometryDrag.objectType ===
            "zone" &&
            geometryDrag.edge !== null
        ) {
            resizeZoneGeometry(
                trade,
                geometryDrag.zoneId,
                geometryDrag.edge,
                original,
                current
            );
        }

        if (
            geometryDrag.mode ===
            "setup-resize"
        ) {
            resizeTradeSetupHorizontally(
                trade,
                geometryDrag.edge,
                original.trade,
                current
            );
        }

        if (
            geometryDrag.mode === "level"
        ) {
            const level =
                geometryDrag.level;

            const price =
                geometryYToPrice(
                    current.y
                );

            if (
                level &&
                price != null &&
                trade.trade_geometry?.[level]
            ) {
                trade.trade_geometry[level].price =
                    price;

                applyTradeRRMode(
                    trade,
                    level,
                    original.trade
                );
            }
        }

        renderLearningGeometry();

        updateGeometrySummary();

        return true;
    }

    function translateGeometryRect(
        rect,
        start,
        current
    ) {
        if (!rect) {
            return null;
        }

        const deltaX =
            current.x -
            start.x;

        const deltaPrice =
            current.price -
            start.price;

        const startX =
            geometryTimeToX(rect.start_time);

        const endX =
            geometryTimeToX(rect.end_time);

        if (
            startX == null ||
            endX == null
        ) {
            return null;
        }

        const newStartTime =
            geometryXToTime(
                startX + deltaX
            );

        const newEndTime =
            geometryXToTime(
                endX + deltaX
            );

        if (
            newStartTime == null ||
            newEndTime == null
        ) {
            return null;
        }

        return {
            start_time: newStartTime,
            end_time: newEndTime,

            low:
                rect.low +
                deltaPrice,

            high:
                rect.high +
                deltaPrice
        };
    }

    function moveTradeSetup(
        target,
        original,
        start,
        current
    ) {
        if (!target || !original) {
            return;
        }

        const deltaX =
            current.x -
            start.x;

        const deltaPrice =
            current.price -
            start.price;

        const originalStartX =
            geometryTimeToX(
                original.start_time
            );

        const originalEndX =
            geometryTimeToX(
                original.end_time
            );

        if (
            originalStartX == null ||
            originalEndX == null
        ) {
            return;
        }

        const newStartTime =
            geometryXToTime(
                originalStartX + deltaX
            );

        const newEndTime =
            geometryXToTime(
                originalEndX + deltaX
            );

        if (
            newStartTime == null ||
            newEndTime == null
        ) {
            return;
        }

        target.start_time =
            newStartTime;

        target.end_time =
            newEndTime;

        [
            "breakout",
            "entry",
            "sl",
            "tp"
        ].forEach(key => {
            if (!original[key]) {
                return;
            }

            const originalX =
                geometryTimeToX(
                    original[key].time
                );

            let newTime =
                original[key].time;

            if (originalX != null) {
                const candidate =
                    geometryXToTime(
                        originalX + deltaX
                    );

                if (candidate != null) {
                    newTime = candidate;
                }
            }

            target[key].time =
                newTime;

            target[key].price =
                original[key].price +
                deltaPrice;
        });
    }

    function resizeZoneGeometry(
        trade,
        zoneId,
        edge,
        original,
        current
    ) {
        const source =
            original.zones.find(
                zone =>
                    String(zone.id) ===
                    String(zoneId)
            );

        if (
            !source ||
            !source.geometry
        ) {
            return;
        }

        const result = {
            ...source.geometry
        };

        if (edge === "left") {
            result.start_time =
                Math.min(
                    current.time,
                    source.geometry.end_time - 1
                );
        }

        if (edge === "right") {
            result.end_time =
                Math.max(
                    current.time,
                    source.geometry.start_time + 1
                );
        }

        if (edge === "top") {
            result.high =
                Math.max(
                    current.price,
                    source.geometry.low
                );
        }

        if (edge === "bottom") {
            result.low =
                Math.min(
                    current.price,
                    source.geometry.high
                );
        }

        const target =
            trade.zones.find(
                zone =>
                    String(zone.id) ===
                    String(zoneId)
            );

        if (target) {
            target.geometry =
                result;
        }
    }

    function resizeTradeSetupHorizontally(
        trade,
        edge,
        original,
        current
    ) {
        if (
            !trade ||
            !original
        ) {
            return;
        }

        const oldStart =
            original.start_time;

        const oldEnd =
            original.end_time;

        const oldSpan =
            oldEnd -
            oldStart;

        if (oldSpan <= 0) {
            return;
        }

        let newStart =
            oldStart;

        let newEnd =
            oldEnd;

        if (edge === "left") {
            newStart =
                Math.min(
                    current.time,
                    oldEnd - 1
                );
        }

        if (edge === "right") {
            newEnd =
                Math.max(
                    current.time,
                    oldStart + 1
                );
        }

        trade.trade_geometry.start_time =
            newStart;

        trade.trade_geometry.end_time =
            newEnd;

        const newSpan =
            newEnd -
            newStart;

        [
            "breakout",
            "entry",
            "sl",
            "tp"
        ].forEach(key => {
            const point =
                original[key];

            if (!point) {
                return;
            }

            const ratio =
                clamp(
                    (point.time - oldStart) /
                        oldSpan,
                    0,
                    1
                );

            trade.trade_geometry[key].time =
                newStart +
                newSpan * ratio;

            trade.trade_geometry[key].price =
                point.price;
        });
    }

    function commitFreeMoveInteraction(event) {
        if (
            !geometryDrag ||
            geometryDrag.mode !== "move"
        ) {
            return;
        }

        const trade =
            getSelectedLearningTrade();

        if (!trade) {
            return;
        }

        const chartRect =
            chartContainer.getBoundingClientRect();

        const releaseY =
            clamp(
                event.clientY -
                    chartRect.top,
                0,
                chartRect.height
            );

        /*
        * ---------------------------------------------------------
        * VERTIKÁLNY POSUN
        *
        * Počas dragovania bol objekt voľný.
        * Až teraz zistíme nový PRICE offset.
        * ---------------------------------------------------------
        */

        const startPrice =
            Number(
                geometryDrag.start.price
            );

        const releasePrice =
            geometryYToPrice(
                releaseY
            );

        const deltaPrice =
            releasePrice != null
                ? Number(releasePrice) -
                startPrice
                : 0;

        /*
        * ---------------------------------------------------------
        * HORIZONTÁLNY POSUN
        *
        * Najprv zistíme nový pixel X.
        * Potom nájdeme najbližšiu sviečku.
        * ---------------------------------------------------------
        */

        const deltaX =
            event.clientX -
            geometryDrag.startClientX;

        const original =
            geometryDrag.original;

        function getSnappedTimeShift(
            originalStartTime
        ) {
            if (
                originalStartTime == null
            ) {
                return 0;
            }

            const originalStartX =
                geometryTimeToX(
                    originalStartTime
                );

            if (originalStartX == null) {
                return 0;
            }

            const draggedStartX =
                originalStartX +
                deltaX;

            const snappedTime =
                snapXToCandleTime(
                    draggedStartX
                );

            if (snappedTime == null) {
                return 0;
            }

            return (
                Number(snappedTime) -
                Number(originalStartTime)
            );
        }

        /*
        * =========================================================
        * MOVE ZONE
        * =========================================================
        */

        if (
            geometryDrag.objectType ===
            "zone" &&
            geometryDrag.zoneId !== null
        ) {
            const source =
                original.zones.find(
                    zone =>
                        String(zone.id) ===
                        String(
                            geometryDrag.zoneId
                        )
                );

            const target =
                trade.zones.find(
                    zone =>
                        String(zone.id) ===
                        String(
                            geometryDrag.zoneId
                        )
                );

            if (
                source &&
                source.geometry &&
                target
            ) {
                const timeShift =
                    getSnappedTimeShift(
                        source.geometry.start_time
                    );

                target.geometry = {
                    ...source.geometry,

                    start_time:
                        Number(
                            source.geometry.start_time
                        ) +
                        timeShift,

                    end_time:
                        Number(
                            source.geometry.end_time
                        ) +
                        timeShift,

                    low:
                        Number(
                            source.geometry.low
                        ) +
                        deltaPrice,

                    high:
                        Number(
                            source.geometry.high
                        ) +
                        deltaPrice
                };
            }
        }

        /*
        * =========================================================
        * MOVE WHOLE TRADE SETUP
        * =========================================================
        */

        if (
            geometryDrag.objectType ===
            "trade" &&
            original.trade
        ) {
            const source =
                original.trade;

            ensureLearningGeometry(
                trade
            );

            const timeShift =
                getSnappedTimeShift(
                    source.start_time
                );

            trade.trade_geometry.start_time =
                Number(
                    source.start_time
                ) +
                timeShift;

            trade.trade_geometry.end_time =
                Number(
                    source.end_time
                ) +
                timeShift;

            [
                "breakout",
                "entry",
                "sl",
                "tp"
            ].forEach(key => {
                if (!source[key]) {
                    return;
                }

                trade.trade_geometry[key] = {
                    time:
                        Number(
                            source[key].time
                        ) +
                        timeShift,

                    price:
                        Number(
                            source[key].price
                        ) +
                        deltaPrice
                };
            });

            /*
            * Prepočet smeru obchodu.
            */
            syncTradeDirectionFromSetup(
                trade
            );

            /*
            * Synchronizácia jednoduchých
            * trade hodnôt z geometrie.
            */
            trade.breakout =
                trade.trade_geometry
                    .breakout?.price ??
                null;

            trade.entry =
                trade.trade_geometry
                    .entry?.price ??
                null;

            trade.sl =
                trade.trade_geometry
                    .sl?.price ??
                null;

            trade.tp =
                trade.trade_geometry
                    .tp?.price ??
                null;
        }
    }

    function canRenderCurrentGeometryMove() {
        if (!geometryDrag || geometryDrag.mode !== "move") {
            return true;
        }

        const trade = getSelectedLearningTrade();

        if (!trade) {
            return false;
        }

        // ZONE
        if (
            geometryDrag.objectType === "zone" &&
            geometryDrag.zoneId !== null
        ) {
            const zone = (trade.zones || []).find(
                item =>
                    String(item.id) ===
                    String(geometryDrag.zoneId)
            );

            if (!zone?.geometry) {
                return false;
            }

            const rect =
                normalizeGeometryRect(
                    zone.geometry
                );

            if (!rect) {
                return false;
            }

            const x1 =
                geometryTimeToX(rect.start_time);

            const x2 =
                geometryTimeToX(rect.end_time);

            const y1 =
                geometryPriceToY(rect.high);

            const y2 =
                geometryPriceToY(rect.low);

            return (
                x1 != null &&
                x2 != null &&
                y1 != null &&
                y2 != null
            );
        }

        // WHOLE SETUP
        if (
            geometryDrag.objectType === "trade" &&
            trade.trade_geometry
        ) {
            const geometry =
                trade.trade_geometry;

            if (
                geometry.start_time == null ||
                geometry.end_time == null
            ) {
                return false;
            }

            const x1 =
                geometryTimeToX(
                    geometry.start_time
                );

            const x2 =
                geometryTimeToX(
                    geometry.end_time
                );

            const prices = [
                geometry.breakout?.price,
                geometry.entry?.price,
                geometry.sl?.price,
                geometry.tp?.price
            ]
                .filter(
                    price =>
                        price != null &&
                        Number.isFinite(
                            Number(price)
                        )
                )
                .map(Number);

            if (!prices.length) {
                return false;
            }

            const high =
                Math.max(...prices);

            const low =
                Math.min(...prices);

            const y1 =
                geometryPriceToY(high);

            const y2 =
                geometryPriceToY(low);

            return (
                x1 != null &&
                x2 != null &&
                y1 != null &&
                y2 != null
            );
        }

        return true;
    }

    function finishGeometryObjectInteraction(
        event
    ) {
        if (!geometryDrag) {
            return;
        }

        /*
        * Pri MOVE najprv commitneme
        * voľný pixelový pohyb do grafových hodnôt.
        */
        if (
            geometryDrag.mode === "move"
        ) {
            commitFreeMoveInteraction(
                event
            );
        }

        if (
            geometryDrag.mode === "move" &&
            !canRenderCurrentGeometryMove()
        ) {
            console.warn(
                "🟡 [MOVE WAIT] coordinate ešte nie je dostupná. Objekt zostáva prichytený."
            );

            return;
        }

        overlay
            .querySelectorAll(
                "[data-geometry-group]"
            )
            .forEach(element => {
                element.style.translate = "";
                element.style.transition = "";
            });

        geometryDrag = null;

        document.body.style.userSelect = "";

        document.body.style.cursor = "";

        renderLearningGeometry();

        updateGeometrySummary();
    }
    
    function syncLearningTradePriceField(inputElement, fieldName) {
        if (selectedLearningTradeId === null) {
            return;
        }

        const trade = learningTrades.find(
            item => item.id === selectedLearningTradeId
        );

        if (!trade) {
            return;
        }

        trade[fieldName] =
            inputElement.value.trim() === ""
                ? null
                : Number(inputElement.value);

        renderLearningGeometry();
    }

    function centerLearningChartOnAnchor() {

        if (!pendingAnchorCenter) {
            return;
        }

        if (
            selectedLearningTradeId === null
        ) {
            return;
        }

        const trade =
            learningTrades.find(
                item =>
                    item.id ===
                    selectedLearningTradeId
            );

        if (
            !trade ||
            !trade.anchor ||
            trade.anchor.time == null
        ) {
            pendingAnchorCenter = false;
            return;
        }

        const anchorTime =
            Number(trade.anchor.time);

        if (!Number.isFinite(anchorTime)) {
            pendingAnchorCenter = false;
            return;
        }

        const visibleRange =
            chart.timeScale()
                .getVisibleRange();

        let span = 60 * 60 * 24;

        if (
            visibleRange &&
            Number.isFinite(Number(visibleRange.from)) &&
            Number.isFinite(Number(visibleRange.to))
        ) {

            span =
                Number(visibleRange.to) -
                Number(visibleRange.from);

            if (
                !Number.isFinite(span) ||
                span <= 0
            ) {
                span = 60 * 60 * 24;
            }
        }

        const halfSpan =
            span / 2;

        chart.timeScale().setVisibleRange({
            from: anchorTime - halfSpan,
            to: anchorTime + halfSpan
        });

        pendingAnchorCenter = false;

        requestAnimationFrame(
            renderLearningAnchor
        );
    }

    function openLearningMode() {
        learningMode = true;

        sidebar.classList.remove("collapsed");

        resultsView.style.display = "none";
        learningView.style.display = "block";

        learningMainView.style.display = "block";
        learningListView.style.display = "none";
        learningDetailView.style.display = "none";

        sidebarToggle.querySelector("img").src = "images/left.png";

        updateLearningHeader();

        renderLearningTradeList();

        if (
            socket &&
            socket.readyState === WebSocket.OPEN
        ) {
            socket.send(JSON.stringify({
                action: "learning_load"
            }));
        }
    }

    function openLearningListView() {
        learningMainView.style.display = "none";
        learningListView.style.display = "block";
        learningDetailView.style.display = "none";

        renderLearningTradeList();
    }

    function closeLearningMode() {
        learningMode = false;

        learningView.style.display = "none";
        resultsView.style.display = "block";

        sidebar.classList.remove("collapsed");

        sidebarToggle.querySelector("img").src = "images/right.png";
    }

    const logoutButton =
        document.getElementById("logoutButton");

    if (WEB_MODE) {
    logoutButton.style.display = "block";
    } else {
        logoutButton.style.display = "none";
    }

    logoutButton.addEventListener("click", () => {
        const dashboardUrl =
            window.SPECTLINE_DASHBOARD_URL ||
            "https://www.spectline.com/dashboard";

        window.location.href = dashboardUrl;
    });

    function connect() {

    document.addEventListener(
        "mousemove",
        event => {
            if (!geometryDrag) {
                return;
            }

            if (
                geometryDrag.mode === "draw"
            ) {
                return;
            }

            const updated =
                updateGeometryObjectInteraction(
                    event
                );

            if (updated) {
                event.preventDefault();
                event.stopPropagation();
            }
        },
        true
    );

    document.addEventListener(
        "mouseup",
        event => {
            if (!geometryDrag) {
                return;
            }

            if (
                geometryDrag.mode === "draw"
            ) {
                return;
            }

            finishGeometryObjectInteraction(
                event
            );
        },
        true
    );   

    chartContainer.addEventListener(
        "mousedown",
        (event) => {

            if (isMobileTouchDevice()) {
                return;
            }

            if (
                event.button !== 0 ||
                !activeGeometryTool
            ) {
                return;
            }

            const started =
                beginGeometryDraw(event);

            if (started) {
                event.preventDefault();
                event.stopPropagation();
            }
        },
        true
    );


    chartContainer.addEventListener(
        "mousemove",
        (event) => {

            if (isMobileTouchDevice()) {
                return;
            }

            if (!geometryDrag) {
                return;
            }

            const updated =
                updateGeometryDraw(event);

            if (updated) {
                event.preventDefault();
                event.stopPropagation();
            }
        },
        true
    );


    chartContainer.addEventListener(
        "mouseup",
        (event) => {

            if (isMobileTouchDevice()) {
                return;
            }

            if (!geometryDrag) {
                return;
            }

            const finished =
                finishGeometryDraw(event);

            if (finished) {
                event.preventDefault();
                event.stopPropagation();
            }
        },
        true
    );

    chartContainer.addEventListener("mousedown", (event) => {
        if (isMobileTouchDevice()) {
            return;
        }

        if (event.button !== 0) {
            return;
        }

        chartMouseDownX = event.clientX;
        chartMouseDownY = event.clientY;
        chartWasDragged = false;
    });

    chartContainer.addEventListener("mousemove", (event) => {
         if (isMobileTouchDevice()) {
            return;
        }

        if ((event.buttons & 1) !== 1) {
            return;
        }

        const dx = event.clientX - chartMouseDownX;
        const dy = event.clientY - chartMouseDownY;

        if (Math.abs(dx) > 4 || Math.abs(dy) > 4) {
            chartWasDragged = true;
        }
    });
   
    chartContainer.addEventListener("click", (event) => {

        if (
            isMobileTouchDevice() &&
            shouldSuppressMobileClick(event)
        ) {
            return;
        }

        if (chartWasDragged) {
            chartWasDragged = false;
            return;
        }

        const insidePlot = isInsideMainPlot(
            event.clientX,
            event.clientY
        );

        if (activeAnchorPicker) {
            if (!insidePlot) {
                return;
            }

            pickAnchorFromChart(event);
            return;
        }

        if (activePriceTarget) {
            if (!insidePlot) {
                return;
            }

            pickPriceFromChart(event);
            return;
        }

        const target = event.target instanceof Element
            ? event.target
            : null;

        const directlyClickedTrade =
            target?.closest(".trade-box");

        if (directlyClickedTrade) {
            return;
        }


        if (isMobileTouchDevice() && insidePlot) {
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

        if (!selectedTradeId) {
            return;
        }

        closeTradeDetail();

        requestAnimationFrame(renderOverlay);
    });

    chartContainer.addEventListener("wheel", () => {
        chartWasDragged = true;

        setTimeout(() => {
            chartWasDragged = false;
        }, 100);
    }, { passive: true });
   
    /* ZOOM THE CHART WHEN THE MOUSE IS OVER A DRAWN OBJECT */
    overlay.addEventListener("wheel", (event) => {
        const target = event.target;

        if (!(target instanceof Element)) {
            return;
        }

        const interactiveObject = target.closest(
            "#overlay .trade-box, " +
            "#overlay .learning-geometry-rect, " +
            "#overlay .learning-geometry-resize-handle, " +
            "#overlay .learning-trade-setup-edge, " +
            "#overlay .learning-setup-level, " +
            "#overlay .learning-zone-rect"
        );

        // Do not interfere with normal chart interaction
        // outside interactive overlay objects.
        if (!interactiveObject) {
            return;
        }

        const timeScale = chart.timeScale();
        const visibleRange = timeScale.getVisibleLogicalRange();

        if (!visibleRange || event.deltaY === 0) {
            return;
        }

        const currentSpan =
            visibleRange.to - visibleRange.from;

        if (!Number.isFinite(currentSpan) || currentSpan <= 0) {
            return;
        }

        // Find the horizontal chart coordinate below the cursor.
        const chartRect = chartContainer.getBoundingClientRect();

        const localX =
            event.clientX -
            chartRect.left -
            getChartPaneLeftOffset();

        const x = Math.max(
            0,
            Math.min(timeScale.width(), localX)
        );

        const logicalAtCursor =
            timeScale.coordinateToLogical(x);

        if (
            logicalAtCursor == null ||
            !Number.isFinite(logicalAtCursor)
        ) {
            return;
        }

        // Wheel up = zoom in; wheel down = zoom out.
        const zoomFactor = Math.exp(
            Math.max(
                -0.5,
                Math.min(0.5, event.deltaY * 0.0015)
            )
        );

        const totalBars = Math.max(
            10,
            Number(lastData?.h1?.length ?? 50000)
        );

        const newSpan = Math.max(
            5,
            Math.min(totalBars, currentSpan * zoomFactor)
        );

        // Keep the candle under the cursor at the same screen position.
        const cursorRatio = Math.max(
            0,
            Math.min(
                1,
                (logicalAtCursor - visibleRange.from) / currentSpan
            )
        );

        const newFrom =
            logicalAtCursor - cursorRatio * newSpan;

        const newTo = newFrom + newSpan;

        event.preventDefault();

        timeScale.setVisibleLogicalRange({
            from: newFrom,
            to: newTo
        });

    }, { passive: false });
        
        const backendBase =
            window.location.protocol === "file:"
                ? "http://127.0.0.1:8000"
                : `${window.location.protocol}//${window.location.host}`;

        const wsProtocol = backendBase.startsWith("https:") ? "wss" : "ws";
        const wsBase = backendBase.replace(/^https?:/, wsProtocol + ":");

        socket = new WebSocket(
            `${wsBase}/ws?ticket=${encodeURIComponent(
                tradingbotLaunchTicket || ""
            )}`
        );

        socket.onopen = () => {
            document.getElementById("status").textContent = "Connected";
        };

        socket.onmessage = event => {
            try {
                const data = JSON.parse(event.data);

                updateManualAnalysisButton(data);

                if (
                    data.analysis_phase === "training_completed" ||
                    data.analysis_phase === "testing_completed" ||
                    data.analysis_phase === "training_error" ||
                    data.analysis_phase === "testing_error" ||
                    data.analysis_phase === "manual_analysis_completed" ||
                    data.analysis_phase === "manual_analysis_error"
                ) {
                    trainingButton.disabled = false;
                    testingButton.disabled = false;
                    manualAnalysisButton.disabled = false;
                    learningAddButton.disabled = false;
                }

                if (data?.type === "trade_detail") {
                    if (!data.success) {
                        console.error(
                            "Trade detail error:",
                            data.error
                        );
                        return;
                    }

                    selectedTradeId = Number(data.trade_id);
                    selectedTradeDetail = data.trade;

                    renderTradeDetailPanel();

                    requestAnimationFrame(() => {
                        renderOverlay();
                    });

                    return;
                }

                if (
                    (
                        data?.learning_action === "loaded" ||
                        data?.learning_action === "saved" ||
                        data?.learning_action === "deleted"
                    ) &&
                    Array.isArray(data.learning_trades)
                ) {
                    learningTrades =
                        data.learning_trades;

                    nextLearningTradeId =
                        learningTrades.reduce(
                            (maxId, trade) =>
                                Math.max(
                                    maxId,
                                    Number(trade.id) || 0
                                ),
                            0
                        ) + 1;

                    renderLearningTradeList();

                    if (
                        data.ready === true &&
                        Array.isArray(data.h1) &&
                        Array.isArray(data.line)
                    ) {
                        lastData = data;
                    }

                    requestAnimationFrame(() => {
                        renderOverlay();
                    });

                    return;
                }

                render(data);

            } catch (error) {
                console.error(error);
            }
        };

        socket.onclose = () => {
            document.getElementById("status").textContent = "Disconnected";
            setTimeout(connect, 1500);
        };

        socket.onerror = () => {
            document.getElementById("status").textContent = "Connection error";
        };
    }

    chart.applyOptions({
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight
    });

    const sidebar = document.getElementById("sidebar");
    const sidebarToggle = document.getElementById("sidebarToggle");
   
    /* =========================================================
    MOBILE SIDEBAR + TOUCH GESTURES
    ========================================================= */

    let mobileGesture = null;
    let mobileLongPressTimer = null;
    let mobileClickSuppression = null;
    let mobileClickSuppressionTimer = null;

    const MOBILE_LONG_PRESS_MS = 380;
    const MOBILE_MOVE_THRESHOLD = 9;

    function clearMobileLongPressTimer() {
        if (mobileLongPressTimer !== null) {
            clearTimeout(mobileLongPressTimer);
            mobileLongPressTimer = null;
        }
    }

    function rememberMobileDragClick(point) {
        if (mobileClickSuppressionTimer !== null) {
            clearTimeout(mobileClickSuppressionTimer);
        }

        mobileClickSuppression = {
            x: point.clientX,
            y: point.clientY,
            expiresAt: Date.now() + 350
        };

        mobileClickSuppressionTimer = setTimeout(() => {
            mobileClickSuppression = null;
            mobileClickSuppressionTimer = null;
        }, 350);
    }

    function shouldSuppressMobileClick(event) {
        if (!mobileClickSuppression) {
            return false;
        }

        if (Date.now() > mobileClickSuppression.expiresAt) {
            mobileClickSuppression = null;
            return false;
        }

        const distance = Math.hypot(
            event.clientX - mobileClickSuppression.x,
            event.clientY - mobileClickSuppression.y
        );

        if (distance > 35) {
            return false;
        }

        mobileClickSuppression = null;

        if (mobileClickSuppressionTimer !== null) {
            clearTimeout(mobileClickSuppressionTimer);
            mobileClickSuppressionTimer = null;
        }

        return true;
    }

    function findOverlayElementAtPoint(selector, clientX, clientY) {
        let bestElement = null;
        let bestZIndex = -Infinity;

        for (const element of overlay.querySelectorAll(selector)) {
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

    function findEditableGeometryAtPoint(clientX, clientY) {
        const HIT_RADIUS = 18;

        let bestHandle = null;
        let bestHandleScore = Infinity;

        const handles = overlay.querySelectorAll(
            ".learning-setup-level, " +
            ".learning-trade-setup-edge, " +
            ".learning-geometry-resize-handle"
        );

        for (const element of handles) {
            if (
                element.classList.contains("manual-setup-view-only")
            ) {
                continue;
            }

            const rect = element.getBoundingClientRect();

            let distance;
            let priority = 0;

            if (
                element.classList.contains("learning-setup-level")
            ) {
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
                const dx =
                    clientX < rect.left
                        ? rect.left - clientX
                        : clientX > rect.right
                            ? clientX - rect.right
                            : 0;

                const dy =
                    clientY < rect.top
                        ? rect.top - clientY
                        : clientY > rect.bottom
                            ? clientY - rect.bottom
                            : 0;

                distance = Math.hypot(dx, dy);

                priority =
                    element.classList.contains(
                        "learning-trade-setup-edge"
                    )
                        ? 1
                        : 2;
            }

            if (distance > HIT_RADIUS) {
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

    function makeTouchMouseEvent(touch, currentTarget = null) {
        return {
            button: 0,
            clientX: touch.clientX,
            clientY: touch.clientY,
            target: currentTarget,
            currentTarget,
            preventDefault() {},
            stopPropagation() {}
        };
    }

    function cancelMobileGestureAction() {
        if (!mobileGesture?.startedAction || !geometryDrag) {
            return;
        }

        if (geometryDrag.mode === "draw") {
            geometryDrag = null;

            overlay
                .querySelectorAll(".learning-geometry-preview")
                .forEach(element => element.remove());
        } else {
            const trade = getSelectedLearningTrade();

            if (trade && geometryDrag.original) {
                trade.zones = geometryDrag.original.zones;
                trade.trade_geometry = geometryDrag.original.trade;
            }

            geometryDrag = null;

            renderLearningGeometry();
            updateGeometrySummary();
        }

        document.body.style.userSelect = "";
        document.body.style.cursor = "";
    }

    /* Outside tap collapses the panel.
    Opening/closing only changes the visual state. */
    document.addEventListener(
        "pointerdown",
        event => {
            if (
                !isMobileTouchDevice() ||
                event.pointerType !== "touch"
            ) {
                return;
            }

            const target = event.target instanceof Element
                ? event.target
                : null;

            if (!target) {
                return;
            }

            if (sidebar.contains(target)) {
                if (
                    sidebar.classList.contains("collapsed") &&
                    !target.closest("#sidebarToggle")
                ) {
                    sidebar.classList.remove("collapsed");

                  sidebarToggle.querySelector("img").src =
                    learningMode
                        ? "images/left.png"
                        : "images/right.png";
                }

                return;
            }

            sidebar.classList.add("collapsed");
            sidebarToggle.querySelector("img").src = "images/left.png";
        },
        true
    );

    /* Touch start: arm a long press only for an active drawing
    tool or an editable geometry object. */
    chartPanel.addEventListener(
        "touchstart",
        event => {
            if (!isMobileTouchDevice()) {
                return;
            }

            if (event.touches.length > 1) {
                if (mobileGesture?.startedAction) {
                    cancelMobileGestureAction();
                }

                clearMobileLongPressTimer();
                mobileGesture = null;
                return;
            }

            const touch = event.touches[0];

            if (!isInsideMainPlot(touch.clientX, touch.clientY)) {
                clearMobileLongPressTimer();
                mobileGesture = null;
                return;
            }

            const target = event.target instanceof Element
                ? event.target
                : null;

            if (
                target?.closest(
                    "button, input, select, textarea, a, " +
                    "#tradeDetailPanel, #mt5-offline-popup, #loading-screen"
                )
            ) {
                clearMobileLongPressTimer();
                mobileGesture = null;
                return;
            }

            clearMobileLongPressTimer();

            const pickerActive =
                activeAnchorPicker || Boolean(activePriceTarget);

            const drawingTool = pickerActive
                ? null
                : activeGeometryTool;

            const editableTarget =
                pickerActive || drawingTool
                    ? null
                    : findEditableGeometryAtPoint(
                        touch.clientX,
                        touch.clientY
                    );

            mobileGesture = {
                startX: touch.clientX,
                startY: touch.clientY,
                mode: activeAnchorPicker
                    ? "anchor-hold"
                    : pickerActive
                        ? "picker"
                        : drawingTool
                            ? "draw"
                            : editableTarget
                                ? "edit"
                                : "pan",
                tool: drawingTool,
                editableTarget,
                moved: false,
                longPressReady: false,
                startedAction: false
            };

            if (activeAnchorPicker || drawingTool || editableTarget) {
                mobileLongPressTimer = setTimeout(() => {
                    if (mobileGesture && !mobileGesture.moved) {
                        mobileGesture.longPressReady = true;

                        if (
                            mobileGesture.mode === "edit" &&
                            typeof navigator.vibrate === "function"
                        ) {
                            navigator.vibrate(35);
                        }
                    }
                }, MOBILE_LONG_PRESS_MS);
            }
        },
        { capture: true, passive: true }
    );

    /* Touch move: ordinary movement pans the chart.
    After a long press, movement draws or edits an object. */
    chartPanel.addEventListener(
        "touchmove",
        event => {
            if (!isMobileTouchDevice()) {
                return;
            }

            if (event.touches.length > 1) {
                if (mobileGesture?.startedAction) {
                    cancelMobileGestureAction();
                }

                clearMobileLongPressTimer();
                mobileGesture = null;
                return;
            }

            if (!mobileGesture) {
                return;
            }

            const touch = event.touches[0];

            const distance = Math.hypot(
                touch.clientX - mobileGesture.startX,
                touch.clientY - mobileGesture.startY
            );

            if (mobileGesture.mode === "anchor-hold") {

                if (!mobileGesture.longPressReady) {
                    if (distance > MOBILE_MOVE_THRESHOLD) {
                        mobileGesture.moved = true;
                        clearMobileLongPressTimer();
                    }

                    return;
                }

                if (event.cancelable) {
                    event.preventDefault();
                }

                event.stopPropagation();
                return;
            }

            if (
                !mobileGesture.startedAction &&
                !mobileGesture.longPressReady
            ) {
                if (distance > MOBILE_MOVE_THRESHOLD) {
                    mobileGesture.moved = true;
                    clearMobileLongPressTimer();
                }

                return;
            }

            if (!mobileGesture.startedAction) {
                if (geometryDrag) {
                    return;
                }

                const startTouch = {
                    clientX: mobileGesture.startX,
                    clientY: mobileGesture.startY
                };

                const startEvent = makeTouchMouseEvent(
                    startTouch,
                    mobileGesture.editableTarget
                );

                if (mobileGesture.tool) {
                    beginGeometryDraw(startEvent);
                } else if (mobileGesture.editableTarget) {
                    beginGeometryObjectInteraction(startEvent);
                }

                if (!geometryDrag) {
                    mobileGesture.moved = true;
                    clearMobileLongPressTimer();
                    return;
                }

                mobileGesture.startedAction = true;
            }

            if (mobileGesture.startedAction && geometryDrag) {
                const currentEvent = makeTouchMouseEvent(
                    touch,
                    mobileGesture.editableTarget
                );

                if (geometryDrag.mode === "draw") {
                    updateGeometryDraw(currentEvent);
                } else {
                    updateGeometryObjectInteraction(currentEvent);
                }

                if (event.cancelable) {
                    event.preventDefault();
                }

                event.stopPropagation();
            }
        },
        { capture: true, passive: false }
    );

    /* Touch end: finish a drawing/edit, or allow the normal tap
    to reach the chart click handler. */
    chartPanel.addEventListener(
        "touchend",
        event => {
            if (!isMobileTouchDevice() || !mobileGesture) {
                return;
            }

            clearMobileLongPressTimer();

            const touch = event.changedTouches[0];

            if (mobileGesture.mode === "anchor-hold") {

                if (
                    mobileGesture.longPressReady &&
                    !mobileGesture.moved
                ) {
                    pickAnchorFromChart({
                        clientX: touch.clientX,
                        clientY: touch.clientY
                    });
                }

                if (event.cancelable) {
                    event.preventDefault();
                }

                rememberMobileDragClick(touch);
                mobileGesture = null;
                return;
            }

            if (mobileGesture.startedAction && geometryDrag) {
                const endEvent = makeTouchMouseEvent(
                    touch,
                    mobileGesture.editableTarget
                );

                if (geometryDrag.mode === "draw") {
                    finishGeometryDraw(endEvent);
                } else {
                    finishGeometryObjectInteraction(endEvent);
                }

                if (event.cancelable) {
                    event.preventDefault();
                }

                rememberMobileDragClick(touch);
            } else if (

                mobileGesture.mode === "edit" &&
                mobileGesture.longPressReady

            ) {

                if (event.cancelable) {
                    event.preventDefault();
                }

                rememberMobileDragClick(touch);

            } else if (mobileGesture.moved) {

                rememberMobileDragClick(touch);
                
            }

            mobileGesture = null;
        },
        { capture: true, passive: false }
    );

    /* Cancel an unfinished gesture if the browser cancels touch input. */
    chartPanel.addEventListener(
        "touchcancel",
        () => {
            if (!isMobileTouchDevice()) {
                return;
            }

            cancelMobileGestureAction();
            clearMobileLongPressTimer();
            mobileGesture = null;
        },
        { capture: true, passive: true }
    );


    const marketSelect = document.getElementById("marketSelect");

    const timeframeSelect = document.getElementById("timeframeSelect");

    marketSelect.addEventListener("change", () => {
        const symbol = marketSelect.value;

        if (!socket || socket.readyState !== WebSocket.OPEN) {
            return;
        }

        pendingSessionId = `${symbol}|${timeframeSelect.value}`;
        resetChartView();

        document.getElementById("status").textContent =
            `Loading ${symbol} ${timeframeSelect.value}...`;

        socket.send(JSON.stringify({
            action: "symbol",
            symbol: symbol
        }));

        updateLearningHeader();
    });

    timeframeSelect.addEventListener("change", () => {

        if (
            WEB_MODE &&
            !WEB_ALLOWED_TIMEFRAMES.includes(
                timeframeSelect.value
            )
        ) {
            timeframeSelect.value = "M15";
            return;
        }

        const timeframe =
            timeframeSelect.value;

        if (
            !socket ||
            socket.readyState !== WebSocket.OPEN
        ) {
            return;
        }

        pendingSessionId =
            `${marketSelect.value}|${timeframe}`;

        resetChartView();

        document.getElementById("status").textContent =
            `Loading ${marketSelect.value} ${timeframe}...`;

        if (
            learningMode &&
            learningDetailView.style.display !== "none" &&
            selectedLearningTradeId !== null
        ) {
            pendingAnchorCenter = true;
        }

        socket.send(JSON.stringify({
            action: "timeframe",
            timeframe: timeframe
        }));

        updateLearningHeader();
    });

    learningButton.addEventListener("click", () => {
        openLearningListView();
    });

    resultsTabButton.addEventListener("click", () => {
        closeLearningMode();
    });

    learningTabButton.addEventListener("click", () => {
        openLearningMode();
    });

    updateDataButton.addEventListener("click", () => {
        if (!socket || socket.readyState !== WebSocket.OPEN) {
            document.getElementById("status").textContent =
                "Backend is not connected.";
            return;
        }

        updateDataButton.disabled = true;

        document.getElementById("status").textContent =
            "Updating market data...";

        socket.send(JSON.stringify({
            action: "update_data"
        }));

        setTimeout(() => {
            updateDataButton.disabled = false;
        }, 1000);
    });

    learningAddButton.addEventListener("click", () => {

        createLearningTrade();

    });

    manualAnalysisButton.addEventListener("click", () => {

        if (
            !socket ||
            socket.readyState !== WebSocket.OPEN
        ) {
            document.getElementById("status").textContent =
                "Backend is not connected.";
            return;
        }

        manualAnalysisActive = true;

        manualAnalysisButton.disabled = true;
        manualAnalysisButton.textContent =
            "ANALYSIS 0%";

        learningAddButton.disabled = true;
        trainingButton.disabled = true;
        testingButton.disabled = true;

        document.getElementById("status").textContent =
            "Analyzing manually entered trades...";

        socket.send(JSON.stringify({
            action: "manual_analysis"
        }));
    });

    learningSaveButton.addEventListener("click", () => {

        saveLearningTrade();

    });

    const drawTradeSetupButton =
        document.getElementById(
            "drawTradeSetupButton"
        );

    function setupLearningAccordion(
        expandButton
    ) {
        if (!expandButton) {
            return;
        }

        expandButton.addEventListener(
            "click",
            event => {
                event.preventDefault();
                event.stopPropagation();

                const item =
                    expandButton.closest(
                        ".learning-accordion-item"
                    );

                if (!item) {
                    return;
                }

                const isOpen =
                    item.classList.toggle("open");

                expandButton.textContent =
                    isOpen ? "▲" : "▼";
            }
        );
    }

    setupLearningAccordion(
        expandAnchorButton
    );

    setupLearningAccordion(
        expandTradeSetupButton
    );

    drawTradeSetupButton.addEventListener(
        "click",
        () => {
            setGeometryTool("trade");

            drawTradeSetupButton.classList.add(
                "selected"
            );
        }
    );

    addZoneButton.addEventListener(
        "click",
        event => {
            event.preventDefault();
            event.stopPropagation();

            const isVisible =
                zoneTypeMenu.style.display !== "none";

            zoneTypeMenu.style.display =
                isVisible
                    ? "none"
                    : "flex";
        }
    );

    addSupportButton.addEventListener(
        "click",
        event => {
            event.preventDefault();
            event.stopPropagation();

            zoneTypeMenu.style.display =
                "none";

            setGeometryTool(
                "zone",
                "SUPPORT"
            );
        }
    );

    addResistanceButton.addEventListener(
        "click",
        event => {
            event.preventDefault();
            event.stopPropagation();

            zoneTypeMenu.style.display =
                "none";

            setGeometryTool(
                "zone",
                "RESISTANCE"
            );
        }
    );

    deleteAnchorButton.addEventListener(
        "click",
        event => {
            event.preventDefault();
            event.stopPropagation();

            const trade =
                getSelectedLearningTrade();

            if (!trade) {
                return;
            }

            trade.anchor = null;

            learningAnchorTime.value = "";
            learningAnchorPrice.value = "";

            pendingAnchorCenter = false;

            cancelAnchorPicker();

            renderLearningAnchor();
        }
    );

    learningRRModeButtons.forEach(button => {
        button.addEventListener(
            "click",
            event => {
                event.preventDefault();
                event.stopPropagation();

                const trade =
                    getSelectedLearningTrade();

                if (!trade) {
                    return;
                }

                const mode =
                    button.dataset.rrMode || "OFF";

                trade.rr_mode = mode;

                learningRRModeButtons.forEach(item => {
                    item.classList.toggle(
                        "active",
                        item === button
                    );
                });
            }
        );
    });

    deleteTradeSetupButton.addEventListener(
        "click",
        event => {
            event.preventDefault();
            event.stopPropagation();

            const trade =
                getSelectedLearningTrade();

            if (!trade) {
                return;
            }

            trade.trade_geometry = {
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

            trade.breakout = null;
            trade.entry = null;
            trade.sl = null;
            trade.tp = null;

            drawTradeSetupButton?.classList.remove("selected");

            renderLearningGeometry();
            updateGeometrySummary();
        }
    );

    learningDeleteButton.addEventListener("click", () => {
        if (selectedLearningTradeId === null) {
            return;
        }

        if (
            !socket ||
            socket.readyState !== WebSocket.OPEN
        ) {
            console.error("Backend is not connected.");
            return;
        }

        socket.send(JSON.stringify({
            action: "learning_delete",
            setup_id: selectedLearningTradeId
        }));

        selectedLearningTradeId = null;
        selectedLearningTradeIsNew = false;

        learningDetailView.style.display = "none";
        learningListView.style.display = "block";
    });

    document
        .querySelectorAll(".learning-price-picker")
        
        
        .forEach(button => {

            button.addEventListener(
                "click",
                (event) => {

                    event.preventDefault();
                    event.stopPropagation();

                    const targetId =
                        button.dataset.priceTarget;

                    activatePricePicker(
                        targetId,
                        button
                    );

                }
            );

        });

    learningAnchorPicker.addEventListener(
        "click",
        (event) => {

            event.preventDefault();
            event.stopPropagation();

            activateAnchorPicker();

        }
    );    
   
    sidebarToggle.addEventListener("click", () => {

        // On mobile, tapping the collapsed rail opens it
        // without leaving the currently selected section.
        if (
            isMobileTouchDevice() &&
            sidebar.classList.contains("collapsed")
        ) {
            sidebar.classList.remove("collapsed");

            sidebarToggle.querySelector("img").src =
                learningMode
                    ? "images/left.png"
                    : "images/right.png";

            return;
        }

        // Preserve the existing desktop navigation behavior.
        if (learningMode) {

            if (
                learningDetailView.style.display !== "none"
            ) {
                closeLearningTradeDetail();
                return;
            }

            if (
                learningListView.style.display !== "none"
            ) {
                learningListView.style.display = "none";
                learningDetailView.style.display = "none";
                learningMainView.style.display = "block";
                return;
            }

            closeLearningMode();
            return;
        }

        sidebar.classList.toggle("collapsed");

      sidebarToggle.querySelector("img").src =
        sidebar.classList.contains("collapsed")
            ? "images/left.png"
            : "images/right.png";
    });

    loadWebMode().then(() => {
        connect();
    });