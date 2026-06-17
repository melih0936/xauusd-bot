"""
News Filter — blocks trading around high-impact USD economic events.
Gold reacts hard to these; trading straight into them is one of the
easiest ways for any bot (or human) to take an unexpected loss.

COVERS (recurring, predictable, generated/hardcoded below):
  - NFP             — Non-Farm Payrolls, first Friday of every month, 8:30 ET
  - Jobless Claims   — every Thursday, 8:30 ET
  - FOMC             — rate decision, 14:00 ET  (confirmed 2026 dates)
  - CPI              — inflation report, 8:30 ET (confirmed dates only — update
                       from https://www.bls.gov/schedule/ as new ones are published)

DOES NOT COVER:
  Surprise Fed speeches, emergency meetings, geopolitical shocks, breaking
  news. A static calendar can only catch what's scheduled in advance —
  it has no way to know about something that hasn't been announced yet.
"""

from datetime import datetime, timedelta

import config


# ── Confirmed FOMC rate decisions (14:00 ET on the 2nd day of each meeting) ──
FOMC_DATES_ET = [
    datetime(2026, 1, 28, 14, 0),
    datetime(2026, 3, 18, 14, 0),
    datetime(2026, 4, 29, 14, 0),
    datetime(2026, 6, 17, 14, 0),
    datetime(2026, 7, 29, 14, 0),
    datetime(2026, 9, 16, 14, 0),
    datetime(2026, 10, 28, 14, 0),
    datetime(2026, 12, 9, 14, 0),
]

# ── Confirmed CPI releases (8:30 ET) — ADD MORE as BLS publishes them ──
CPI_DATES_ET = [
    datetime(2026, 5, 12, 8, 30),
    datetime(2026, 6, 10, 8, 30),
    datetime(2026, 7, 14, 8, 30),
    datetime(2026, 8, 12, 8, 30),
    # Sept 2026 onward: check https://www.bls.gov/schedule/ and add here
]


# ──────────────────────────────────────────────────────────────────
# Recurring events generated algorithmically (no maintenance needed)
# ──────────────────────────────────────────────────────────────────

def _first_friday(year, month):
    d = datetime(year, month, 1)
    while d.weekday() != 4:          # 4 = Friday
        d += timedelta(days=1)
    return d.replace(hour=8, minute=30)

def _all_thursdays(year, month):
    d = datetime(year, month, 1)
    out = []
    while d.month == month:
        if d.weekday() == 3:          # 3 = Thursday
            out.append(d.replace(hour=8, minute=30))
        d += timedelta(days=1)
    return out

def _generate_recurring(years=(2025, 2026, 2027)):
    events = []
    for year in years:
        for month in range(1, 13):
            events.append(("NFP", _first_friday(year, month)))
            for th in _all_thursdays(year, month):
                events.append(("Jobless Claims", th))
    return events


def _build_event_list():
    events = list(_generate_recurring())
    events += [("FOMC", dt) for dt in FOMC_DATES_ET]
    events += [("CPI",  dt) for dt in CPI_DATES_ET]
    offset = timedelta(hours=config.SERVER_OFFSET_FROM_ET_HOURS)
    return [(name, dt + offset) for name, dt in events]   # convert ET → broker server time


_EVENTS = _build_event_list()


def is_news_blackout(current_time, window_minutes=None):
    """
    current_time: naive datetime in the SAME frame as your MT5 bar timestamps
                  (broker server time — not raw UTC).
    Returns (is_blackout: bool, event_name: str|None, minutes_away: float|None)
    """
    window = timedelta(minutes=window_minutes or config.NEWS_BLACKOUT_MINUTES)
    for name, event_time in _EVENTS:
        delta = current_time - event_time
        if abs(delta) <= window:
            return True, name, delta.total_seconds() / 60
    return False, None, None
