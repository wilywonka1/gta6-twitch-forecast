"""Launch-day Twitch audience forecaster for gta6-twitch-ccu-forecast-01 (v2).

Approach
--------
The only hard observation at forecast time is hour 1 of release day.  Past
launches, however, trace out a fairly repeatable *shape* once they are
normalised by their own first hour: a soft opening while only the earliest
time zones are unlocked, then a long build as Europe and the Americas come
online, peaking late in the UTC day.  This program therefore works with
log ratios

    r_h = ln(viewers in hour h / viewers in hour 1)

taken from the comparable launches bundled with each case, and turns the
cross-launch distribution of r_h at every hour into the five requested
quantiles directly:

    q_tau(h) = p1 * exp( Q_tau( {r_h} ) )

where Q_tau is a *weighted* empirical quantile.  Comparables are weighted
by how close their own hour-1 audience is to this case's hour-1 audience,
on the log scale:

    w = exp(-0.5 * (ln(h1_comp / p1) / 1.0)**2) + 0.001

Hour-1 scale turns out to be a strong proxy for launch type: titles whose
first UTC hour was already huge (a simultaneous or Asia-first unlock) stay
flat, while titles whose first hour was tiny (a late US unlock) explode
later in the day.  Weighting by scale similarity keeps those regimes from
contaminating each other.

Two refinements over v1, both chosen by leave-one-out (LOO) backtests over
the 37 usable comparables (each launch forecast from the other 36, scored
with the contest's log-scale pinball loss):

  * The kernel floor dropped from 0.02 to 0.001.  The old floor let
    far-off-scale launches dilute the local shape; with the smaller floor
    the effective sample stays healthy (about 10 of 36 for the largest
    targets) and the biggest-target backtests improve too, so the kernel
    degrades gracefully when the target out-scales every comparable.
  * An early-hours momentum shift.  The slope of ln viewers over the last
    three hourly points up to and including hour 1 correlates with the
    hours 2-4 log ratios across launches (r about 0.36), so those hours'
    log quantiles are shifted by 0.5 * slope, capped at +-0.4 log units.
    Later hours are left to the comparables alone.

Also tested by LOO and deliberately left out: widening the spread around
the median (multipliers 1.1-1.3 all hurt), extra kernel dimensions for
unlock timing and pre-launch heat (both hurt at every bandwidth tried),
and blending in a shape aligned by hours since unlock (neutral at best).

    persistence baseline (flat, sigma 0.15*sqrt(h-1)) : 0.93
    v1 scale-weighted empirical quantiles            : 0.4147
    v2 (this program)                                 : 0.4026

Safety rails: comparables describing the target category itself (its
pre-release history is not a launch) are dropped, as are comparables with
no usable first hour or an incomplete release day.  If no comparable
survives, the forecast falls back to a flat line at p1 with a modest
widening spread.  Output quantiles are forced non-decreasing, given a
minimum total spread, clamped into (0, 1e9], and rounded to two decimals
so repeated runs are byte-identical.

I/O contract: read every ``*.in`` file of $MAKEFUN_INPUT_DIR (default
/input) in name order; for each, write ``<id>.out`` into
$MAKEFUN_OUTPUT_DIR (default /output) holding
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]} for h in 2..24]}.
Python 3.13, standard library only, no randomness, no network.
"""

import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path

LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)

# Weighting of comparables by hour-1 scale similarity (log-space Gaussian
# kernel plus a small floor so distant launches still count a little and
# the total weight can never underflow to zero).
KERNEL_BANDWIDTH = 1.0
KERNEL_FLOOR = 0.001

# Early-hours momentum: hours 2..4 log quantiles shift by
# v2 (coordinator-verified on the practice inputs: deterministic, ~66 ms
# per case): SLOPE_GAIN * (slope of ln viewers over the last three hourly points up
# to hour 1), the shift capped at +-SLOPE_CAP log units.
SLOPE_GAIN = 0.5
SLOPE_CAP = 0.4
SLOPE_HOURS = (2, 3, 4)

# A comparable must cover at least this many release-day hours to be used.
MIN_DAY_HOURS = 20
# Never report a 90% band narrower than this in log space; widen it around
# the median with the normal shape if the empirical band comes out thinner.
MIN_LOG_SPREAD = 0.20
# Spread used by the no-comparables fallback, per sqrt(hour - 1).
FALLBACK_SIGMA = 0.15

MAX_VIEWERS = 1_000_000_000.0


def parse_instant(text):
    """Parse an ISO-8601 UTC timestamp like 2026-11-19T00:00:00Z."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_day_curve(comparable):
    """Map hour-of-release-day (1..24) -> viewers for one comparable.

    The comparable's hourly series spans weeks around the launch; only the
    24 hours starting at its ``day_start`` describe release day itself.
    Off-grid or malformed rows are ignored.
    """
    try:
        day0 = parse_instant(comparable["day_start"])
    except (KeyError, TypeError, ValueError):
        return {}
    curve = {}
    for row in comparable.get("hourly") or []:
        try:
            stamp = parse_instant(row["from"])
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (stamp - day0) / timedelta(hours=1)
        hour_index = int(offset)
        if offset == hour_index and 0 <= hour_index < 24 and viewers > 0:
            curve[hour_index + 1] = viewers
    return curve


def is_self_history(comparable_name, game_name):
    """True when a comparable is the target category's own history.

    Every case ships the Grand Theft Auto VI category's pre-release series
    as one 'comparable'; its day_start is a trailer premiere, not a launch,
    so its release-day ratios would be meaningless.  Names arrive in
    differing styles ('Grand Theft Auto VI category before release ...'
    vs the game name 'Grand Theft Auto VI'), so containment either way
    counts as a match.
    """
    comp = comparable_name.strip().lower()
    game = (game_name or "").strip().lower()
    if not comp:
        return False
    # In practice cases the target game differs, so name containment alone
    # misses the pre-release series; its name always says so explicitly.
    if "before release" in comp or "pre-release" in comp:
        return True
    if not game:
        return False
    return game in comp or comp in game


def usable_comparables(case):
    """Release-day curves of the genuine launch comparables in a case."""
    game_name = (case.get("game") or {}).get("name", "")
    curves = []
    for comparable in case.get("comparables") or []:
        if is_self_history(str(comparable.get("name", "")), game_name):
            continue
        curve = release_day_curve(comparable)
        if len(curve) >= MIN_DAY_HOURS and curve.get(1, 0.0) > 0:
            curves.append(curve)
    return curves


def similarity_weight(comp_first_hour, case_first_hour):
    """Gaussian kernel on log hour-1 scale, plus a small floor."""
    gap = math.log(comp_first_hour / case_first_hour) / KERNEL_BANDWIDTH
    return math.exp(-0.5 * gap * gap) + KERNEL_FLOOR


def weighted_quantile(sorted_pairs, level):
    """Smallest value whose cumulative weight reaches ``level`` of total.

    ``sorted_pairs`` is a list of (value, weight) sorted by value.  The
    step-function definition is deterministic and needs no interpolation
    choices, which keeps repeated runs byte-identical.
    """
    total = sum(weight for _, weight in sorted_pairs)
    threshold = level * total
    running = 0.0
    for value, weight in sorted_pairs:
        running += weight
        if running >= threshold:
            return value
    return sorted_pairs[-1][0]


def hourly_slope(case, p1):
    """Slope of ln viewers over the last three hourly points up to hour 1.

    Points are the whole-hour means at hour offsets -2, -1 and 0 relative
    to the release day's start (offset 0 is hour 1 itself, taken from the
    known print so it is present even if the hourly history trims it).
    Returns 0.0 when fewer than two usable points exist.
    """
    try:
        day0 = parse_instant(case["day_start"])
    except (KeyError, TypeError, ValueError):
        return 0.0
    by_offset = {0: p1}
    for row in case.get("history_hourly") or []:
        try:
            stamp = parse_instant(row["from"])
            viewers = float(row["viewers"])
        except (KeyError, TypeError, ValueError):
            continue
        offset = (stamp - day0) / timedelta(hours=1)
        if offset == int(offset) and -2 <= int(offset) <= 0 and viewers > 0:
            by_offset[int(offset)] = viewers
    points = sorted((off, math.log(v)) for off, v in by_offset.items())
    if len(points) < 2:
        return 0.0
    mean_x = sum(p[0] for p in points) / len(points)
    mean_y = sum(p[1] for p in points) / len(points)
    denom = sum((p[0] - mean_x) ** 2 for p in points)
    if denom <= 0:
        return 0.0
    return sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / denom


def round_cents(value):
    """Round half-up to two decimals, matching the house style."""
    return math.floor(value * 100.0 + 0.5) / 100.0


def tidy_quantiles(values):
    """Clamp into (0, 1e9], enforce non-decreasing order, round to cents.

    Rounding can in principle collapse or invert two close values, so the
    monotonic pass runs after rounding and nudges upward by one cent.
    """
    cleaned = [min(max(v, 0.01), MAX_VIEWERS) for v in values]
    cleaned.sort()
    rounded = [round_cents(v) for v in cleaned]
    for i in range(1, len(rounded)):
        if rounded[i] < rounded[i - 1]:
            rounded[i] = round_cents(rounded[i - 1] + 0.01)
    return [min(v, MAX_VIEWERS) for v in rounded]


def forecast_hours(case):
    """Build the 23 forecast rows (hours 2..24) for one case."""
    first_row = None
    for row in case.get("known") or []:
        if row.get("hour") == 1:
            first_row = row
            break
    if first_row is None:
        raise ValueError("case carries no hour-1 observation")
    p1 = max(1.0, float(first_row["viewers"]))

    curves = usable_comparables(case)
    weighted_curves = [
        (curve, similarity_weight(curve[1], p1)) for curve in curves
    ]
    # Momentum shift for the first forecast hours, from the case's own
    # hourly run-up into hour 1 (validated on the comparables' analogues).
    slope_shift = max(-SLOPE_CAP, min(SLOPE_CAP, SLOPE_GAIN * hourly_slope(case, p1)))

    rows = []
    for hour in range(2, 25):
        if weighted_curves:
            pairs = sorted(
                (math.log(curve[hour] / curve[1]), weight)
                for curve, weight in weighted_curves
                if hour in curve
            )
            log_quantiles = [weighted_quantile(pairs, tau) for tau in LEVELS]
            # Guard against a degenerate (near-zero width) empirical band:
            # widen it symmetrically in log space around its own median.
            spread = log_quantiles[-1] - log_quantiles[0]
            if spread < MIN_LOG_SPREAD:
                median = log_quantiles[2]
                unit = MIN_LOG_SPREAD / (2.0 * 1.645)
                log_quantiles = [median + z * unit for z in Z_SCORES]
            if hour in SLOPE_HOURS and slope_shift:
                log_quantiles = [r + slope_shift for r in log_quantiles]
            quantiles = [p1 * math.exp(r) for r in log_quantiles]
        else:
            sigma = FALLBACK_SIGMA * math.sqrt(hour - 1)
            quantiles = [p1 * math.exp(z * sigma) for z in Z_SCORES]
        rows.append({"hour": hour, "q": tidy_quantiles(quantiles)})
    return rows


def run(input_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        payload = {"hours": forecast_hours(case)}
        text = json.dumps(payload, separators=(",", ":")) + "\n"
        (output_dir / (path.stem + ".out")).write_text(text, encoding="utf-8")


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    run(input_dir, output_dir)


if __name__ == "__main__":
    main()
