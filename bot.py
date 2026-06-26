#!/usr/bin/env python3
"""
SOL 9EMA / 9SMA Crossover Alert Bot
=====================================

Watches SOLUSDT on Binance across 3m, 5m and 15m timeframes.
On each timeframe, independently checks for a 9 EMA / 9 SMA crossover:

    BUY  signal -> 9 EMA crosses ABOVE the 9 SMA  (bullish crossover)
    SELL signal -> 9 EMA crosses BELOW the 9 SMA  (bearish crossover)

Each Telegram alert also includes, purely as extra context (it does NOT
gate the signal):
    - RSI(28) value
    - 13 EMA value and whether price/9EMA is currently above or below it

Dependencies are auto-installed on first run (requests).

------------------------------------------------------------------
SECURITY NOTE (read this):
Your Telegram bot token was shared in plain text in chat. Treat it as
compromised. Go to @BotFather on Telegram -> /mybots -> select your bot
-> API Token -> Revoke/Regenerate, then put the NEW token below (or,
better, set it as an environment variable instead of hardcoding it,
see CONFIG section).
------------------------------------------------------------------

Deployment note (Railway):
This script runs as an infinite loop with a sleep, which is exactly the
shape Railway wants for a long-running worker process. Just set this as
your Start Command (e.g. `python sol_ema_sma_bot.py`) in a Railway
service, and optionally move the token/chat id into Railway's
Variables tab instead of hardcoding them (recommended).
"""

import sys
import subprocess
import importlib


# ---------------------------------------------------------------------------
# 0. AUTO-INSTALL DEPENDENCIES
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = ["requests"]


def ensure_dependencies():
    for pkg in REQUIRED_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"[setup] '{pkg}' not found, installing...")
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", pkg]
            )
            print(f"[setup] '{pkg}' installed.")


ensure_dependencies()

import os
import time
import logging
from collections import deque
from datetime import datetime, timezone

import requests


# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
# Prefer environment variables (Railway -> Variables tab). Falls back to the
# hardcoded values below ONLY if the env var isn't set, so this still runs
# standalone on your own machine.

TELEGRAM_TOKEN = os.environ.get(
    "TELEGRAM_TOKEN", "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"
)
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "1950462171")

SYMBOL = os.environ.get("SYMBOL", "SOLUSDT")
TIMEFRAMES = ["3m", "5m", "15m"]   # Binance interval strings

EMA_FAST_PERIOD = 9      # 9 EMA
SMA_PERIOD = 9           # 9 SMA
EMA_TREND_PERIOD = 13    # 13 EMA (informational)
RSI_PERIOD = 28          # RSI(28) (informational)

POLL_SECONDS = int(os.environ.get("POLL_SECONDS", "15"))  # how often to check
KLINES_LIMIT = 200        # candles to fetch (plenty for warm-up of all indicators)

BINANCE_BASE_URL = "https://api.binance.com"
KLINES_ENDPOINT = "/api/v3/klines"

# ---------------------------------------------------------------------------
# 2. LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("sol_bot")


# ---------------------------------------------------------------------------
# 3. INDICATOR MATH (pure python, no extra deps needed)
# ---------------------------------------------------------------------------
def sma(values, period):
    """Simple Moving Average series. Returns list aligned to `values`
    (first `period-1` entries are None)."""
    out = [None] * len(values)
    if len(values) < period:
        return out
    window_sum = sum(values[:period])
    out[period - 1] = window_sum / period
    for i in range(period, len(values)):
        window_sum += values[i] - values[i - period]
        out[i] = window_sum / period
    return out


def ema(values, period):
    """Exponential Moving Average series. Seeded with SMA of the first
    `period` values, standard convention. Returns list aligned to
    `values` (first `period-1` entries are None)."""
    out = [None] * len(values)
    if len(values) < period:
        return out
    k = 2 / (period + 1)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1 - k)
        out[i] = prev
    return out


def rsi(values, period):
    """Wilder's RSI. Returns list aligned to `values`."""
    out = [None] * len(values)
    if len(values) < period + 1:
        return out

    gains = []
    losses = []
    for i in range(1, period + 1):
        change = values[i] - values[i - 1]
        gains.append(max(change, 0))
        losses.append(max(-change, 0))

    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    def calc_rsi(avg_gain, avg_loss):
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    out[period] = calc_rsi(avg_gain, avg_loss)

    for i in range(period + 1, len(values)):
        change = values[i] - values[i - 1]
        gain = max(change, 0)
        loss = max(-change, 0)
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        out[i] = calc_rsi(avg_gain, avg_loss)

    return out


# ---------------------------------------------------------------------------
# 4. DATA FETCH
# ---------------------------------------------------------------------------
def fetch_klines(symbol, interval, limit=KLINES_LIMIT):
    """Fetch candles from Binance public REST API. Returns list of dicts
    with at least 'close_time' and 'close' (float), oldest -> newest.
    The most recent candle returned by Binance is usually still FORMING
    (not closed yet) -- we keep that in mind in the signal logic."""
    url = BINANCE_BASE_URL + KLINES_ENDPOINT
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    raw = resp.json()

    candles = []
    for k in raw:
        candles.append(
            {
                "open_time": k[0],
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
                "close_time": k[6],
                "is_closed": True,  # corrected below for the last candle
            }
        )
    if candles:
        candles[-1]["is_closed"] = False  # last candle is still forming
    return candles


# ---------------------------------------------------------------------------
# 5. TELEGRAM
# ---------------------------------------------------------------------------
def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram token/chat id not set, skipping send.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, data=payload, timeout=10)
        if resp.status_code != 200:
            log.error(f"Telegram send failed: {resp.status_code} {resp.text}")
    except requests.RequestException as e:
        log.error(f"Telegram send exception: {e}")


# ---------------------------------------------------------------------------
# 6. SIGNAL ENGINE (per timeframe, tracks last crossover state to avoid
#    spamming the same signal repeatedly)
# ---------------------------------------------------------------------------
class TimeframeTracker:
    """Holds last-known crossover relationship (ema9 vs sma9) for one
    timeframe so we only alert on the actual cross, not every poll."""

    def __init__(self, symbol, interval):
        self.symbol = symbol
        self.interval = interval
        self.last_relationship = None  # "above" | "below" | None
        self.last_closed_open_time = None  # avoid reprocessing same candle

    def check(self):
        try:
            candles = fetch_klines(self.symbol, self.interval)
        except Exception as e:
            log.error(f"[{self.interval}] fetch error: {e}")
            return

        # Only evaluate on CLOSED candles to avoid false signals from a
        # half-formed candle repainting the indicators every few seconds.
        closed = [c for c in candles if c["is_closed"]]
        if len(closed) < max(EMA_TREND_PERIOD, RSI_PERIOD + 1, SMA_PERIOD) + 2:
            log.info(f"[{self.interval}] not enough closed candles yet, skipping.")
            return

        latest_closed = closed[-1]
        if self.last_closed_open_time == latest_closed["open_time"]:
            return  # already processed this candle, nothing new
        self.last_closed_open_time = latest_closed["open_time"]

        closes = [c["close"] for c in closed]

        ema9_series = ema(closes, EMA_FAST_PERIOD)
        sma9_series = sma(closes, SMA_PERIOD)
        ema13_series = ema(closes, EMA_TREND_PERIOD)
        rsi28_series = rsi(closes, RSI_PERIOD)

        ema9_now = ema9_series[-1]
        sma9_now = sma9_series[-1]
        ema13_now = ema13_series[-1]
        rsi28_now = rsi28_series[-1]

        if ema9_now is None or sma9_now is None:
            return

        current_relationship = "above" if ema9_now > sma9_now else "below"

        # First time we have a reading: just record it, don't fire an alert
        # (we don't know what happened "before" we started watching).
        if self.last_relationship is None:
            self.last_relationship = current_relationship
            log.info(
                f"[{self.interval}] baseline set: 9EMA is {current_relationship} 9SMA "
                f"(ema9={ema9_now:.4f}, sma9={sma9_now:.4f})"
            )
            return

        crossed_up = self.last_relationship == "below" and current_relationship == "above"
        crossed_down = self.last_relationship == "above" and current_relationship == "below"

        if crossed_up or crossed_down:
            self.fire_alert(
                direction="BUY" if crossed_up else "SELL",
                price=latest_closed["close"],
                ema9=ema9_now,
                sma9=sma9_now,
                ema13=ema13_now,
                rsi28=rsi28_now,
                candle_time=latest_closed["close_time"],
            )

        self.last_relationship = current_relationship

    def fire_alert(self, direction, price, ema9, sma9, ema13, rsi28, candle_time):
        ts = datetime.fromtimestamp(candle_time / 1000, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        emoji = "🟢" if direction == "BUY" else "🔴"
        ema13_status = "N/A"
        if ema13 is not None:
            ema13_status = "Price above 13EMA (uptrend bias)" if price > ema13 else "Price below 13EMA (downtrend bias)"

        rsi_status = "N/A"
        if rsi28 is not None:
            if rsi28 >= 70:
                rsi_status = f"{rsi28:.2f} (overbought)"
            elif rsi28 <= 30:
                rsi_status = f"{rsi28:.2f} (oversold)"
            else:
                rsi_status = f"{rsi28:.2f} (neutral)"

        message = (
            f"{emoji} <b>{direction} SIGNAL</b> — {SYMBOL} [{self.interval}]\n"
            f"9 EMA crossed {'ABOVE' if direction == 'BUY' else 'BELOW'} 9 SMA\n\n"
            f"Price: <b>{price:.4f}</b>\n"
            f"9 EMA: {ema9:.4f}\n"
            f"9 SMA: {sma9:.4f}\n"
            f"13 EMA: {ema13:.4f} -> {ema13_status}\n"
            f"RSI(28): {rsi_status}\n\n"
            f"Candle close: {ts}"
        )

        log.info(f"ALERT [{self.interval}] {direction} @ {price}")
        send_telegram_message(message)


# ---------------------------------------------------------------------------
# 7. MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    log.info(f"Starting SOL 9EMA/9SMA crossover bot for {SYMBOL} on {TIMEFRAMES}")
    log.info(f"Poll interval: {POLL_SECONDS}s | RSI period: {RSI_PERIOD} | 13EMA trend filter shown as info only")

    trackers = [TimeframeTracker(SYMBOL, tf) for tf in TIMEFRAMES]

    send_telegram_message(
        f"✅ SOL crossover bot started.\nSymbol: {SYMBOL}\nTimeframes: {', '.join(TIMEFRAMES)}\n"
        f"Watching for 9EMA/9SMA crossovers."
    )

    while True:
        for tracker in trackers:
            tracker.check()
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Stopped by user.")
    except Exception as e:
        log.exception(f"Fatal error: {e}")
        # Try to notify via Telegram before dying, useful on Railway logs too
        try:
            send_telegram_message(f"⚠️ Bot crashed: {e}")
        except Exception:
            pass
        raise
