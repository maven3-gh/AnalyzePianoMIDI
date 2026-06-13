"""Tests for analyze_midi group-histogram analysis."""

import sys
import tempfile
from pathlib import Path

import mido
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analyze_midi import (
    GROUPS,
    _MIDI_OFFSET,
    _group_of,
    build_histograms,
    format_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_midi(path: str, notes: list[tuple[int, float, float, int]]) -> None:
    """Write a type-0 MIDI file from a list of (pitch, start_s, end_s, vel)."""
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))

    events: list[tuple[float, str, int, int]] = []
    for pitch, start_s, end_s, vel in notes:
        events.append((start_s, "note_on",  pitch, vel))
        events.append((end_s,   "note_off", pitch, 0))
    events.sort()

    ticks_per_s = 480 / 0.5  # 480 ticks/beat, 120 BPM → 0.5 s/beat
    prev_s = 0.0
    for t_s, msg_type, pitch, vel in events:
        delta_ticks = int((t_s - prev_s) * ticks_per_s)
        track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta_ticks))
        prev_s = t_s

    mid.save(path)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_group_assignments():
    # Boundary checks for each group
    for g, (lo, hi) in enumerate(GROUPS):
        assert _group_of(lo + _MIDI_OFFSET) == g
        assert _group_of(hi + _MIDI_OFFSET) == g
    # Off-keyboard notes
    assert _group_of(20) is None   # below A0
    assert _group_of(109) is None  # above C8


def test_all_88_keys_assigned():
    for key in range(1, 89):
        midi = key + _MIDI_OFFSET
        g = _group_of(midi)
        assert g is not None, f"Piano key {key} (MIDI {midi}) has no group"


def test_group_sizes():
    expected = [13, 13, 12, 12, 12, 13, 13]
    for i, (lo, hi) in enumerate(GROUPS):
        assert hi - lo + 1 == expected[i], (
            f"Group {i+1}: expected {expected[i]} keys, got {hi - lo + 1}"
        )


def test_single_note_histogram():
    """One note in group 4 (key 39-50, MIDI 59-70) active for 500 ms."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        # MIDI 60 = C4 = piano key 40 → group 4 (index 3)
        make_midi(f.name, [(60, 0.0, 0.5, 80)])
        _, histograms, note_counts = build_histograms(Path(f.name))

    assert note_counts[3] == 1
    assert histograms[3].get(1, 0) == 500  # 500 ms of 1 simultaneous note
    # All other groups should be empty
    for g in range(len(GROUPS)):
        if g != 3:
            assert not histograms[g]


def test_simultaneous_notes():
    """Two notes in the same group at the same time → count=2 in histogram."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        # MIDI 60 (key 40) and MIDI 62 (key 42) — both in group 4 (index 3)
        # Overlap from 0..0.3 s → 300 ms at count 2
        # Note A alone 0.3..0.5 s → 200 ms at count 1
        make_midi(f.name, [
            (60, 0.0, 0.5, 80),
            (62, 0.0, 0.3, 80),
        ])
        _, histograms, _ = build_histograms(Path(f.name))

    h = histograms[3]
    assert h.get(2, 0) == 300
    assert h.get(1, 0) == 200


def test_missing_file():
    with pytest.raises(SystemExit):
        from analyze_midi import analyze_file
        analyze_file(Path("does_not_exist.mid"))


def test_format_report_runs():
    """format_report should return a non-empty string without raising."""
    dummy_hist = [{1: 1000, 2: 200} if g == 3 else {} for g in range(len(GROUPS))]
    dummy_counts = [0, 0, 0, 5, 0, 0, 0]
    report = format_report(Path("test.mid"), 2.0, dummy_hist, dummy_counts)
    assert "Group 4" in report
    assert "1000" in report
