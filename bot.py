#!/usr/bin/env python3
"""
SOLUSDT RSI Crossover Signal Bot
=================================
Tracks RSI(28) crossovers on 3m and 5m timeframes:
  - Signal 1: RSI(28) × EMA(13) crossover
  - Signal 2: RSI(28) × SMA(50) crossover

Polls Binance on candle close and sends Telegram alerts.
"""

import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg---"
TELEGRAM_CHAT_ID = "1950462171"

SYMBOL     = "SOLUSDT"
BINANCE_BASE = "https://api.binance.com"

# RSI & moving-average settings
RSI_PERIOD = 28
EMA_PERIOD = 13
SMA_PERIOD = 50

# Candles to fetch (must be > SMA_PERIOD + RSI_PERIOD for warm-up)
CANDLE_LIMIT = 200

# Timeframes to monitor: (interval_string, poll_interval_seconds)
TIMEFRAMES = [
    ("3m", 3 * 60),
    ("5m", 5 * 60),
]

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.info("Telegram ✅ sent: %s", message[:80])
            return True
        else:
            log.error("Telegram ❌ error %s: %s", resp.status_code, resp.text)
            return False
    except Exception as e:
        log.error("Telegram exception: %s", e)
        return False


# ─── BINANCE DATA ──────────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    Fetch OHLCV klines from Binance public REST API.
    Returns DataFrame with columns: open_time, open, high, low, close, volume, close_time.
    Only closed candles are included (last candle excluded as it may be live).
    """
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit + 1}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
    except requests.RequestException as e:
        log.error("Binance fetch error: %s", e)
        return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    # Keep only closed candles (drop the last live candle)
    df = df.iloc[:-1].copy()

    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["close"]      = df["close"].astype(float)

    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


# ─── INDICATORS ────────────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's smoothed RSI."""
    delta = series.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)

    # Wilder smoothing (equivalent to EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period).mean()


def detect_crossover(a: pd.Series, b: pd.Series) -> str | None:
    """
    Detect crossover between series a and b at the last two candles.
    Returns 'bullish' if a crossed above b, 'bearish' if a crossed below b,
    else None.
    """
    if len(a) < 2 or a.isna().iloc[-2:].any() or b.isna().iloc[-2:].any():
        return None

    prev_above = a.iloc[-2] > b.iloc[-2]
    curr_above = a.iloc[-1] > b.iloc[-1]

    if not prev_above and curr_above:
        return "bullish"
    if prev_above and not curr_above:
        return "bearish"
    return None


# ─── SIGNAL FORMATTING ─────────────────────────────────────────────────────────
ARROW = {"bullish": "🟢 ▲ BULLISH", "bearish": "🔴 ▼ BEARISH"}

def format_signal(
    timeframe: str,
    signal_type: str,
    crossover: str,
    rsi_val: float,
    ma_val: float,
    candle_time: datetime,
    price: float,
) -> str:
    direction = ARROW[crossover]
    ts = candle_time.strftime("%Y-%m-%d %H:%M UTC")

    return (
        f"<b>⚡ RSI CROSSOVER SIGNAL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 <b>Symbol:</b> {SYMBOL}\n"
        f"⏱ <b>Timeframe:</b> {timeframe}\n"
        f"📊 <b>Signal:</b> {signal_type}\n"
        f"🔀 <b>Direction:</b> {direction}\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>RSI(28):</b> {rsi_val:.2f}\n"
        f"📉 <b>MA Value:</b> {ma_val:.2f}\n"
        f"💵 <b>Price:</b> ${price:.4f}\n"
        f"🕐 <b>Candle Close:</b> {ts}\n"
    )


# ─── PER-TIMEFRAME STATE ───────────────────────────────────────────────────────
class TimeframeTracker:
    """
    Tracks crossover state for a single timeframe.
    Stores last seen candle close time to avoid duplicate signals.
    """
    def __init__(self, interval: str, poll_seconds: int):
        self.interval      = interval
        self.poll_seconds  = poll_seconds
        self.last_candle_time: datetime | None = None
        self.last_ema_cross: str | None = None   # last known ema crossover state
        self.last_sma_cross: str | None = None   # last known sma crossover state

    def check_and_signal(self) -> list[str]:
        """
        Fetch latest closed candles, compute indicators, detect crossovers.
        Returns list of Telegram messages to send (may be empty).
        """
        df = fetch_klines(SYMBOL, self.interval)
        if df.empty or len(df) < SMA_PERIOD + RSI_PERIOD:
            log.warning("[%s] Not enough candles returned.", self.interval)
            return []

        last_close_time = df["close_time"].iloc[-1]

        # Skip if we've already processed this candle
        if self.last_candle_time is not None and last_close_time <= self.last_candle_time:
            log.debug("[%s] No new closed candle yet.", self.interval)
            return []

        self.last_candle_time = last_close_time

        close  = df["close"]
        rsi    = compute_rsi(close)
        ema13  = compute_ema(rsi, EMA_PERIOD)
        sma50  = compute_sma(rsi, SMA_PERIOD)

        latest_rsi   = rsi.iloc[-1]
        latest_ema   = ema13.iloc[-1]
        latest_sma   = sma50.iloc[-1]
        latest_price = close.iloc[-1]
        candle_ts    = last_close_time.to_pydatetime()

        messages = []

        # ── Signal 1: RSI(28) × EMA(13) ────────────────────────────────────
        ema_cross = detect_crossover(rsi, ema13)
        if ema_cross and ema_cross != self.last_ema_cross:
            self.last_ema_cross = ema_cross
            msg = format_signal(
                timeframe   = self.interval,
                signal_type = f"RSI({RSI_PERIOD}) × EMA({EMA_PERIOD})",
                crossover   = ema_cross,
                rsi_val     = latest_rsi,
                ma_val      = latest_ema,
                candle_time = candle_ts,
                price       = latest_price,
            )
            messages.append(msg)
            log.info("[%s] EMA crossover: %s | RSI=%.2f EMA=%.2f",
                     self.interval, ema_cross, latest_rsi, latest_ema)

        # ── Signal 2: RSI(28) × SMA(50) ────────────────────────────────────
        sma_cross = detect_crossover(rsi, sma50)
        if sma_cross and sma_cross != self.last_sma_cross:
            self.last_sma_cross = sma_cross
            msg = format_signal(
                timeframe   = self.interval,
                signal_type = f"RSI({RSI_PERIOD}) × SMA({SMA_PERIOD})",
                crossover   = sma_cross,
                rsi_val     = latest_rsi,
                ma_val      = latest_sma,
                candle_time = candle_ts,
                price       = latest_price,
            )
            messages.append(msg)
            log.info("[%s] SMA crossover: %s | RSI=%.2f SMA=%.2f",
                     self.interval, sma_cross, latest_rsi, latest_sma)

        if not messages:
            log.info(
                "[%s] Candle closed @ %s | RSI=%.2f | EMA13=%.2f | SMA50=%.2f | No crossover",
                self.interval,
                candle_ts.strftime("%H:%M"),
                latest_rsi, latest_ema, latest_sma,
            )

        return messages


# ─── SCHEDULER ─────────────────────────────────────────────────────────────────
def seconds_to_next_close(interval_minutes: int) -> float:
    """
    Returns how many seconds until the next candle of this interval closes.
    Adds a 2-second buffer so the candle is definitely closed on Binance.
    """
    now_ts  = time.time()
    period  = interval_minutes * 60
    elapsed = now_ts % period
    wait    = period - elapsed + 2          # +2s buffer
    return wait


def run_tracker(tracker: TimeframeTracker, interval_minutes: int):
    """
    Blocking loop that sleeps until each candle close, then checks for signals.
    Designed to run in a separate thread.
    """
    import threading
    name = f"[{tracker.interval}]"
    log.info("%s tracker started.", name)

    while True:
        wait = seconds_to_next_close(interval_minutes)
        next_check = datetime.now(timezone.utc).replace(microsecond=0)
        log.info("%s Next candle close check in %.0fs  (~%s UTC)",
                 name, wait, next_check)
        time.sleep(wait)

        try:
            messages = tracker.check_and_signal()
            for msg in messages:
                send_telegram(msg)
        except Exception as e:
            log.error("%s Unhandled error: %s", name, e)


# ─── STARTUP TEST ──────────────────────────────────────────────────────────────
def startup_test():
    """Send a startup ping and do a quick data sanity check."""
    log.info("Running startup checks …")

    # Test Binance connectivity
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/ping", timeout=5)
        r.raise_for_status()
        log.info("Binance API ✅ reachable")
    except Exception as e:
        log.error("Binance API ❌ unreachable: %s", e)

    # Test Telegram
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    startup_msg = (
        f"<b>🤖 RSI Crossover Bot Started</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 Symbol: <b>{SYMBOL}</b>\n"
        f"⏱ Timeframes: <b>3m, 5m</b>\n"
        f"📊 RSI Period: <b>{RSI_PERIOD}</b>\n"
        f"📐 Signals: <b>RSI×EMA({EMA_PERIOD})</b> &amp; <b>RSI×SMA({SMA_PERIOD})</b>\n"
        f"🕐 Started: <b>{now_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Monitoring for crossovers on candle close …"
    )
    send_telegram(startup_msg)

    # Quick indicator sanity check on 3m data
    df = fetch_klines(SYMBOL, "3m", limit=100)
    if not df.empty:
        rsi   = compute_rsi(df["close"])
        ema13 = compute_ema(rsi, EMA_PERIOD)
        sma50 = compute_sma(rsi, SMA_PERIOD)
        log.info(
            "Sanity check 3m | RSI=%.2f | EMA13=%.2f | SMA50=%.2f | Price=%.4f",
            rsi.iloc[-1], ema13.iloc[-1], sma50.iloc[-1], df["close"].iloc[-1],
        )
    else:
        log.warning("Could not fetch 3m data for sanity check.")


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    import threading

    log.info("=" * 55)
    log.info("  SOLUSDT RSI Crossover Bot")
    log.info("  Signals: RSI(28)×EMA(13) and RSI(28)×SMA(50)")
    log.info("  Timeframes: 3m and 5m  |  On candle close")
    log.info("=" * 55)

    startup_test()

    threads = []
    for interval_str, poll_secs in TIMEFRAMES:
        interval_minutes = int(interval_str.replace("m", ""))
        tracker = TimeframeTracker(interval_str, poll_secs)

        t = threading.Thread(
            target=run_tracker,
            args=(tracker, interval_minutes),
            name=f"tracker-{interval_str}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        log.info("Thread started for %s", interval_str)

    log.info("Bot running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Shutting down …")
        send_telegram("🛑 <b>RSI Crossover Bot stopped.</b>")


if __name__ == "__main__":
    main()
