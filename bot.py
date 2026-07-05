#!/usr/bin/env python3
"""
SOLUSDT RSI Crossover Signal Bot
=================================
Tracks RSI(28) crossovers on 3m and 5m timeframes:
  - Signal 1 : RSI(28) × EMA(13) crossover
  - Signal 2 : RSI(28) × SMA(50) crossover

Polls Binance exactly on candle close (with +2 s buffer)
Sends formatted Telegram alerts on every new crossover event.

Warm-up analysis
─────────────────
  RSI(28)          → needs 28 bars before first valid value
  EMA(13) of RSI   → needs 28+13 = 41 bars
  SMA(50) of RSI   → needs 28+50 = 78 bars   ← tightest constraint
  Crossover detect → needs 2 valid MA bars   → minimum 80 candles total

  CANDLE_LIMIT = 300  →  300 − 80 = 220 fully stable signal bars ✅
"""

import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN   = "8349229275:AAGNWV2A0_Pf9LhlwZCczeBoMcUaJL2shFg"
TELEGRAM_CHAT_ID = "1950462171"

SYMBOL       = "SOLUSDT"
BINANCE_BASE = "https://api.binance.com"

# Indicator periods
RSI_PERIOD = 28
EMA_PERIOD = 13
SMA_PERIOD = 50

# Candles fetched per request.
# Must satisfy: CANDLE_LIMIT > RSI_PERIOD + SMA_PERIOD + 2  (= 80 minimum)
# We use 300 → 220 stable bars after warm-up, plenty of headroom.
CANDLE_LIMIT = 300

# Minimum usable candles after warm-up (auto-derived, used as a guard)
MIN_CANDLES_REQUIRED = RSI_PERIOD + SMA_PERIOD + 2   # = 80

# Timeframes: (binance interval string, integer minutes)
TIMEFRAMES = [
    ("3m", 3),
    ("5m", 5),
]

# Seconds added after candle close before fetching (lets Binance finalise)
CANDLE_CLOSE_BUFFER_SECS = 2

# Retry settings for Binance fetch
FETCH_RETRIES    = 3
FETCH_RETRY_WAIT = 5   # seconds between retries

# ─── LOGGING ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── TELEGRAM ──────────────────────────────────────────────────────────────────
def send_telegram(message: str) -> bool:
    """Send a message via Telegram Bot API (HTML parse mode)."""
    url     = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, json=payload, timeout=10)
        if resp.status_code == 200:
            log.info("Telegram ✅  %s", message[:80].replace("\n", " "))
            return True
        log.error("Telegram ❌ %s: %s", resp.status_code, resp.text[:120])
        return False
    except Exception as exc:
        log.error("Telegram exception: %s", exc)
        return False


# ─── BINANCE DATA ──────────────────────────────────────────────────────────────
def fetch_klines(symbol: str, interval: str, limit: int = CANDLE_LIMIT) -> pd.DataFrame:
    """
    Fetch closed OHLCV klines from Binance public REST API.

    Requests limit+1 candles and drops the last one because it is still
    forming (live candle).  Retries up to FETCH_RETRIES times on failure.

    Returns DataFrame columns:
        open_time, close_time, open, high, low, close (float), volume
    Returns empty DataFrame on unrecoverable error.
    """
    url    = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit + 1}

    for attempt in range(1, FETCH_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            raw  = resp.json()
            break
        except requests.RequestException as exc:
            log.warning("Binance fetch attempt %d/%d failed: %s",
                        attempt, FETCH_RETRIES, exc)
            if attempt < FETCH_RETRIES:
                time.sleep(FETCH_RETRY_WAIT)
            else:
                log.error("Binance fetch gave up after %d attempts.", FETCH_RETRIES)
                return pd.DataFrame()

    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    # Drop the still-forming live candle (last row)
    df = df.iloc[:-1].copy()

    df["open_time"]  = pd.to_datetime(df["open_time"],  unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df["close"]      = df["close"].astype(float)

    return df[["open_time", "close_time", "open", "high", "low", "close", "volume"]]


# ─── INDICATORS ────────────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """
    Wilder's Smoothed RSI.
    Uses EWM with alpha = 1/period (identical to TradingView's RSI).
    First `period` values are NaN (insufficient history).
    """
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_ema(series: pd.Series, period: int) -> pd.Series:
    """Standard EMA (span = period).  NaN rows from input propagate."""
    return series.ewm(span=period, adjust=False).mean()


def compute_sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average.  First period−1 values are NaN."""
    return series.rolling(window=period, min_periods=period).mean()


def detect_crossover(a: pd.Series, b: pd.Series) -> str | None:
    """
    Detect a crossover between series `a` and `b` using the last two bars.

    Returns
    -------
    'bullish'  — a was below b on bar[-2], now above b on bar[-1]
    'bearish'  — a was above b on bar[-2], now below b on bar[-1]
    None       — no crossover, or insufficient / NaN data
    """
    if len(a) < 2 or len(b) < 2:
        return None
    if a.iloc[-2:].isna().any() or b.iloc[-2:].isna().any():
        return None

    prev_above = float(a.iloc[-2]) > float(b.iloc[-2])
    curr_above = float(a.iloc[-1]) > float(b.iloc[-1])

    if not prev_above and curr_above:
        return "bullish"
    if prev_above and not curr_above:
        return "bearish"
    return None


# ─── SIGNAL FORMATTING ─────────────────────────────────────────────────────────
_DIRECTION = {"bullish": "🟢 ▲ BULLISH CROSS", "bearish": "🔴 ▼ BEARISH CROSS"}

def format_signal(
    timeframe:   str,
    signal_type: str,
    crossover:   str,
    rsi_val:     float,
    ma_val:      float,
    candle_time: datetime,
    price:       float,
) -> str:
    direction = _DIRECTION[crossover]
    ts        = candle_time.strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"<b>⚡ RSI CROSSOVER SIGNAL</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 <b>Symbol    :</b> {SYMBOL}\n"
        f"⏱ <b>Timeframe :</b> {timeframe}\n"
        f"📊 <b>Signal    :</b> {signal_type}\n"
        f"🔀 <b>Direction :</b> {direction}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📈 <b>RSI({RSI_PERIOD})   :</b> {rsi_val:.2f}\n"
        f"📉 <b>MA value  :</b> {ma_val:.2f}\n"
        f"💵 <b>Price     :</b> ${price:.4f}\n"
        f"🕐 <b>Closed at :</b> {ts}\n"
    )


# ─── PER-TIMEFRAME TRACKER ─────────────────────────────────────────────────────
class TimeframeTracker:
    """
    Manages state for one timeframe (3m or 5m).

    On every call to check_and_signal():
      1. Fetches the latest CANDLE_LIMIT closed candles.
      2. Verifies we have enough data for stable indicator warm-up.
      3. Computes RSI, EMA(RSI), SMA(RSI).
      4. Detects crossovers on the freshly closed bar.
      5. Guards against duplicate signals (same direction on consecutive bars).
      6. Returns formatted Telegram messages for any new crossovers.
    """

    def __init__(self, interval: str):
        self.interval         = interval
        self.last_candle_time: datetime | None = None
        # Track last fired direction per signal to suppress duplicates
        self.last_ema_cross:  str | None = None
        self.last_sma_cross:  str | None = None

    # ── public ─────────────────────────────────────────────────────────────────
    def check_and_signal(self) -> list[str]:
        """
        Run one check cycle.  Returns a (possibly empty) list of Telegram messages.
        """
        df = fetch_klines(SYMBOL, self.interval)

        # ── guard: empty response ───────────────────────────────────────────
        if df.empty:
            log.warning("[%s] Empty response from Binance — skipping.", self.interval)
            return []

        # ── guard: insufficient candles for warm-up ─────────────────────────
        if len(df) < MIN_CANDLES_REQUIRED:
            log.warning(
                "[%s] Only %d candles returned (need ≥ %d) — skipping.",
                self.interval, len(df), MIN_CANDLES_REQUIRED,
            )
            return []

        last_close_time = df["close_time"].iloc[-1]

        # ── guard: already processed this candle ────────────────────────────
        if (self.last_candle_time is not None
                and last_close_time <= self.last_candle_time):
            log.debug("[%s] Candle %s already processed — skipping.",
                      self.interval, last_close_time)
            return []

        self.last_candle_time = last_close_time

        # ── compute indicators ──────────────────────────────────────────────
        close = df["close"]
        rsi   = compute_rsi(close)
        ema13 = compute_ema(rsi, EMA_PERIOD)
        sma50 = compute_sma(rsi, SMA_PERIOD)

        latest_rsi   = float(rsi.iloc[-1])
        latest_ema   = float(ema13.iloc[-1])
        latest_sma   = float(sma50.iloc[-1])
        latest_price = float(close.iloc[-1])
        candle_ts    = last_close_time.to_pydatetime()

        # ── guard: NaN in final values (should not happen with CANDLE_LIMIT=300)
        if any(np.isnan(v) for v in (latest_rsi, latest_ema, latest_sma)):
            log.warning("[%s] NaN in indicator values — skipping candle.", self.interval)
            return []

        messages: list[str] = []

        # ── Signal 1 : RSI(28) × EMA(13) ───────────────────────────────────
        ema_cross = detect_crossover(rsi, ema13)
        if ema_cross is not None and ema_cross != self.last_ema_cross:
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
            log.info("[%s] EMA CROSS %s | RSI=%.2f  EMA13=%.2f  price=%.4f",
                     self.interval, ema_cross.upper(), latest_rsi, latest_ema, latest_price)

        # ── Signal 2 : RSI(28) × SMA(50) ───────────────────────────────────
        sma_cross = detect_crossover(rsi, sma50)
        if sma_cross is not None and sma_cross != self.last_sma_cross:
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
            log.info("[%s] SMA CROSS %s | RSI=%.2f  SMA50=%.2f  price=%.4f",
                     self.interval, sma_cross.upper(), latest_rsi, latest_sma, latest_price)

        # ── status log when no crossover ────────────────────────────────────
        if not messages:
            log.info(
                "[%s] ✓ closed %s | RSI=%.2f | EMA13=%.2f | SMA50=%.2f | $%.4f | no cross",
                self.interval,
                candle_ts.strftime("%H:%M"),
                latest_rsi, latest_ema, latest_sma, latest_price,
            )

        return messages


# ─── SCHEDULER ─────────────────────────────────────────────────────────────────
def seconds_to_next_close(interval_minutes: int) -> float:
    """
    Compute seconds until the next candle of this interval finishes.
    Adds CANDLE_CLOSE_BUFFER_SECS so Binance has time to finalise the bar.
    """
    now_ts  = time.time()
    period  = interval_minutes * 60
    elapsed = now_ts % period
    return period - elapsed + CANDLE_CLOSE_BUFFER_SECS


def run_tracker(tracker: TimeframeTracker, interval_minutes: int) -> None:
    """
    Infinite loop: sleep until next candle close, then check for signals.
    Runs in its own daemon thread.
    """
    tag = f"[{tracker.interval}]"
    log.info("%s thread started (interval = %dm).", tag, interval_minutes)

    while True:
        wait        = seconds_to_next_close(interval_minutes)
        wake_at_utc = datetime.now(timezone.utc).replace(microsecond=0)
        log.info("%s sleeping %.0f s until next close (≈%s UTC)",
                 tag, wait, wake_at_utc)
        time.sleep(wait)

        try:
            messages = tracker.check_and_signal()
            for msg in messages:
                send_telegram(msg)
        except Exception as exc:
            log.error("%s unhandled error: %s", tag, exc, exc_info=True)


# ─── STARTUP ───────────────────────────────────────────────────────────────────
def startup_checks() -> None:
    """
    Verify connectivity, log warm-up stats, and fire a Telegram startup message.
    """
    log.info("Running startup checks …")

    # 1. Binance ping
    try:
        r = requests.get(f"{BINANCE_BASE}/api/v3/ping", timeout=5)
        r.raise_for_status()
        log.info("Binance API ✅ reachable")
    except Exception as exc:
        log.error("Binance API ❌ unreachable: %s", exc)

    # 2. Warm-up summary
    log.info(
        "Warm-up: RSI needs %d bars | EMA needs %d | SMA needs %d | "
        "CANDLE_LIMIT=%d → %d stable bars after warm-up",
        RSI_PERIOD,
        RSI_PERIOD + EMA_PERIOD,
        RSI_PERIOD + SMA_PERIOD,
        CANDLE_LIMIT,
        CANDLE_LIMIT - (RSI_PERIOD + SMA_PERIOD + 2),
    )

    # 3. Live indicator sanity check on 3m
    df = fetch_klines(SYMBOL, "3m", limit=CANDLE_LIMIT)
    if not df.empty and len(df) >= MIN_CANDLES_REQUIRED:
        rsi   = compute_rsi(df["close"])
        ema13 = compute_ema(rsi, EMA_PERIOD)
        sma50 = compute_sma(rsi, SMA_PERIOD)
        rsi_nan  = rsi.isna().sum()
        ema_nan  = ema13.isna().sum()
        sma_nan  = sma50.isna().sum()
        log.info(
            "3m live check | candles=%d | RSI NaN=%d | EMA NaN=%d | SMA NaN=%d",
            len(df), rsi_nan, ema_nan, sma_nan,
        )
        log.info(
            "3m live values | RSI=%.2f | EMA13=%.2f | SMA50=%.2f | price=%.4f",
            float(rsi.iloc[-1]), float(ema13.iloc[-1]),
            float(sma50.iloc[-1]), float(df["close"].iloc[-1]),
        )
    else:
        log.warning("Could not complete 3m sanity check (data: %d rows).", len(df))

    # 4. Telegram startup message
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    send_telegram(
        f"<b>🤖 RSI Crossover Bot — ONLINE</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"🪙 <b>Symbol     :</b> {SYMBOL}\n"
        f"⏱ <b>Timeframes :</b> 3m &amp; 5m\n"
        f"📊 <b>RSI period :</b> {RSI_PERIOD}\n"
        f"📐 <b>Signals    :</b> RSI×EMA({EMA_PERIOD})  &amp;  RSI×SMA({SMA_PERIOD})\n"
        f"📦 <b>Candles    :</b> {CANDLE_LIMIT} fetched → "
        f"{CANDLE_LIMIT - (RSI_PERIOD + SMA_PERIOD + 2)} stable bars\n"
        f"🕐 <b>Started    :</b> {now_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Watching for crossovers on candle close …"
    )


# ─── MAIN ──────────────────────────────────────────────────────────────────────
def main() -> None:
    import threading

    log.info("=" * 60)
    log.info("  SOLUSDT RSI Crossover Bot")
    log.info("  Signals : RSI(%d)×EMA(%d)  and  RSI(%d)×SMA(%d)",
             RSI_PERIOD, EMA_PERIOD, RSI_PERIOD, SMA_PERIOD)
    log.info("  TFs     : 3m, 5m  |  trigger: candle close + %ds buffer",
             CANDLE_CLOSE_BUFFER_SECS)
    log.info("  Candles : %d fetched  |  min required: %d  |  stable: %d",
             CANDLE_LIMIT, MIN_CANDLES_REQUIRED,
             CANDLE_LIMIT - MIN_CANDLES_REQUIRED)
    log.info("=" * 60)

    startup_checks()

    threads: list[threading.Thread] = []
    for interval_str, interval_min in TIMEFRAMES:
        tracker = TimeframeTracker(interval_str)
        t = threading.Thread(
            target=run_tracker,
            args=(tracker, interval_min),
            name=f"tracker-{interval_str}",
            daemon=True,
        )
        t.start()
        threads.append(t)
        log.info("Thread started → %s", interval_str)

    log.info("Bot running. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(30)
    except KeyboardInterrupt:
        log.info("Shutting down …")
        send_telegram("🛑 <b>RSI Crossover Bot stopped.</b>")


if __name__ == "__main__":
    main()
