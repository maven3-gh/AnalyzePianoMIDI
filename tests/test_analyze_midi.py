"""Basic tests for analyze_midi using a synthetic MIDI file."""

import sys
import tempfile
from pathlib import Path

import mido
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analyze_midi import analyze, MidiAnalysis


def make_test_midi(path: str) -> None:
    """Write a minimal single-track MIDI with a C major scale."""
    mid = mido.MidiFile(type=0)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C4–C5
    for pitch in pitches:
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))
    mid.save(path)


def test_analyze_returns_correct_types():
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
        make_test_midi(f.name)
        result = analyze(f.name)
    assert isinstance(result, MidiAnalysis)
    assert result.total_notes == 8
    assert result.unique_pitches == 8
    assert result.tempo_bpm == pytest.approx(120.0, abs=0.1)
    assert result.time_signature == "4/4"
    assert result.pitch_range == (60, 72)


def test_analyze_missing_file():
    with pytest.raises(FileNotFoundError):
        analyze("nonexistent.mid")
