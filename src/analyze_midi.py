"""Piano MIDI group-histogram analysis at 1 ms resolution.

Groups the 88 piano keys (MIDI 21-108) into 7 bands and builds a
histogram per group: how many 1-ms time slots had N notes sounding
simultaneously.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pretty_midi

# ---------------------------------------------------------------------------
# Key groupings — (first_key, last_key) in 1-indexed piano-key numbering.
# Piano key N  ↔  MIDI note (N + 20).   Key 1 = A0 = MIDI 21.
# ---------------------------------------------------------------------------
GROUPS: list[tuple[int, int]] = [
    (1,  13),   # keys  1-13  → MIDI 21-33  (A0  – A1)
    (14, 26),   # keys 14-26  → MIDI 34-46  (Bb1 – Bb2)
    (27, 38),   # keys 27-38  → MIDI 47-58  (B2  – Bb3)
    (39, 50),   # keys 39-50  → MIDI 59-70  (B3  – Bb4)
    (51, 62),   # keys 51-62  → MIDI 71-82  (B4  – Bb5)
    (63, 75),   # keys 63-75  → MIDI 83-95  (B5  – B6)
    (76, 88),   # keys 76-88  → MIDI 96-108 (C7  – C8)
]

_MIDI_OFFSET = 20   # MIDI note = piano_key + _MIDI_OFFSET


def _group_of(midi_note: int) -> int | None:
    """Return 0-based group index for midi_note, or None if off-keyboard."""
    key = midi_note - _MIDI_OFFSET
    if not (1 <= key <= 88):
        return None
    for i, (lo, hi) in enumerate(GROUPS):
        if lo <= key <= hi:
            return i
    return None


def _note_name(midi_note: int) -> str:
    return pretty_midi.note_number_to_name(midi_note)


def _group_header(g: int) -> str:
    lo, hi = GROUPS[g]
    lo_m, hi_m = lo + _MIDI_OFFSET, hi + _MIDI_OFFSET
    return (f"Group {g + 1}  |  keys {lo:>2}-{hi:<2}"
            f"  ({_note_name(lo_m)}-{_note_name(hi_m)}, MIDI {lo_m}-{hi_m})")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def build_histograms(midi_path: Path) -> tuple[float, list[dict[int, int]], list[int]]:
    """
    Scan a MIDI file at 1 ms resolution and build per-group histograms.

    Returns
    -------
    duration_s  : total file duration in seconds
    histograms  : list of 7 dicts  {simultaneous_note_count: ms_count}
    note_counts : total notes found per group
    """
    print(f"\n  Loading '{midi_path.name}' ...", flush=True)
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    duration_s = pm.get_end_time()

    # Collect (start_ms, end_ms) per group
    group_intervals: list[list[tuple[int, int]]] = [[] for _ in GROUPS]

    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            g = _group_of(note.pitch)
            if g is None:
                continue
            s_ms = int(note.start * 1000)
            e_ms = int(note.end   * 1000)
            if e_ms > s_ms:
                group_intervals[g].append((s_ms, e_ms))

    note_counts = [len(iv) for iv in group_intervals]

    print(f"  Duration : {duration_s:.2f} s  ({duration_s / 60:.2f} min)")
    print(f"  Notes    : {sum(note_counts)} total  |  per group: "
          + "  ".join(f"G{i+1}={n}" for i, n in enumerate(note_counts)))
    print(f"\n  Building 1 ms histograms for {len(GROUPS)} groups ...", flush=True)

    histograms: list[dict[int, int]] = []

    for g, intervals in enumerate(group_intervals):
        lo, hi = GROUPS[g]
        lo_m, hi_m = lo + _MIDI_OFFSET, hi + _MIDI_OFFSET
        print(f"    [{g+1}/7]  keys {lo:>2}-{hi:<2}"
              f"  ({_note_name(lo_m)}-{_note_name(hi_m)}) ...",
              end="  ", flush=True)

        # Event sweep: build sorted list of (time_ms, +1/-1)
        events: list[tuple[int, int]] = []
        for s_ms, e_ms in intervals:
            events.append((s_ms, +1))
            events.append((e_ms, -1))
        events.sort()

        hist: dict[int, int] = defaultdict(int)
        active = 0
        prev_t = 0
        i = 0

        while i < len(events):
            t = events[i][0]
            # Accumulate interval [prev_t, t) at current active count
            if t > prev_t and active > 0:
                hist[active] += t - prev_t
            # Consume all events sharing time t
            while i < len(events) and events[i][0] == t:
                active += events[i][1]
                i += 1
            prev_t = t

        histograms.append(dict(hist))

        if hist:
            max_sim = max(hist)
            active_ms = sum(hist.values())
            print(f"max simultaneous = {max_sim},  {active_ms} ms active", flush=True)
        else:
            print("(no activity)", flush=True)

    return duration_s, histograms, note_counts


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_BAR_WIDTH = 42


def _bar(value: int, max_value: int) -> str:
    if max_value == 0 or value == 0:
        return ""
    filled = max(1, round(value / max_value * _BAR_WIDTH))
    return "#" * filled


def format_report(
    midi_path: Path,
    duration_s: float,
    histograms: list[dict[int, int]],
    note_counts: list[int],
) -> str:
    SEP  = "=" * 68
    DASH = "-" * 68
    lines: list[str] = []

    lines += [
        SEP,
        "  Piano MIDI Group Histogram Analysis",
        f"  File     : {midi_path.name}",
        f"  Duration : {duration_s:.2f} s  ({duration_s / 60:.2f} min)",
        f"  Groups   : {len(GROUPS)}   |   Piano keys 1-88  (MIDI 21-108)",
        SEP,
    ]

    for g, hist in enumerate(histograms):
        lo, hi = GROUPS[g]
        n_keys = hi - lo + 1

        lines.append("")
        lines.append(_group_header(g) + f"  |  {note_counts[g]} notes")
        lines.append(DASH)

        if not hist:
            lines.append("  (no activity)")
            continue

        max_ms = max(hist.values())
        lines.append(f"  {'Active':>6} | {'ms count':>9} | Bar  (1 '#' = {max_ms / _BAR_WIDTH:.0f} ms)")
        lines.append(f"  {'------':>6}-+-{'----------':>9}-+-" + "-" * _BAR_WIDTH)

        for count in range(1, n_keys + 1):
            ms_val = hist.get(count, 0)
            bar    = _bar(ms_val, max_ms)
            lines.append(f"  {count:>6} | {ms_val:>9} | {bar}")

        lines.append(f"  {'------':>6}-+-{'----------':>9}-+-" + "-" * _BAR_WIDTH)

    lines += ["", SEP]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------

def analyze_file(midi_path: Path) -> None:
    if not midi_path.exists():
        print(f"[ERROR] File not found: {midi_path}", file=sys.stderr)
        sys.exit(1)

    duration_s, histograms, note_counts = build_histograms(midi_path)

    report = format_report(midi_path, duration_s, histograms, note_counts)
    print("\n" + report)

    out_path = midi_path.with_suffix(".txt")
    out_path.write_text(report, encoding="utf-8")
    print(f"\n  Report saved -> {out_path}\n")


def run_batch(midi_dir: Path) -> None:
    """Analyze all MIDI files in midi_dir that lack a matching .txt result."""
    all_midi = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.MID"))
    pending = [f for f in all_midi if not f.with_suffix(".txt").exists()]

    if not pending:
        print(f"\n  No pending MIDI files in '{midi_dir}' — all results up to date.\n")
        return

    total = len(pending)
    done  = 0

    print(f"\n  Found {total} file(s) to analyze in '{midi_dir}':\n")
    for f in pending:
        print(f"    {f.name}")
    print()

    for idx, midi_path in enumerate(pending, start=1):
        SEP = "=" * 68
        print(SEP)
        print(f"  File {idx}/{total}  |  Session count: {done} analyzed  |  {midi_path.name}")
        print(SEP)

        analyze_file(midi_path)
        done += 1

        if idx < total:
            remaining = total - idx
            print(f"  {done} file(s) analyzed this session.  {remaining} remaining.")
            try:
                answer = input("  Continue to next file? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Stopped.")
                break
            if answer in ("n", "no"):
                print(f"\n  Stopped after {done} file(s). Run again to process the rest.\n")
                break
            print()

    print(f"\n  Session complete — {done} of {total} file(s) analyzed.\n")


def main() -> None:
    if len(sys.argv) >= 2:
        for arg in sys.argv[1:]:
            analyze_file(Path(arg))
    else:
        midi_dir = Path(__file__).parent.parent / "midi_files"
        run_batch(midi_dir)


if __name__ == "__main__":
    main()
