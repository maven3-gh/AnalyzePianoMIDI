"""Piano MIDI analysis using mido and pretty_midi."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import mido
import pretty_midi
import numpy as np


@dataclass
class NoteStats:
    pitch: int
    name: str
    count: int
    avg_velocity: float
    avg_duration_s: float


@dataclass
class MidiAnalysis:
    path: Path
    duration_s: float
    tempo_bpm: float
    time_signature: str
    total_notes: int
    unique_pitches: int
    avg_velocity: float
    pitch_range: tuple[int, int]
    top_notes: list[NoteStats] = field(default_factory=list)

    def print_report(self) -> None:
        print(f"\n{'='*48}")
        print(f"  {self.path.name}")
        print(f"{'='*48}")
        print(f"  Duration      : {self.duration_s:.2f}s")
        print(f"  Tempo         : {self.tempo_bpm:.1f} BPM")
        print(f"  Time sig      : {self.time_signature}")
        print(f"  Total notes   : {self.total_notes}")
        print(f"  Unique pitches: {self.unique_pitches}")
        print(f"  Avg velocity  : {self.avg_velocity:.1f}")
        lo, hi = self.pitch_range
        print(f"  Pitch range   : {pretty_midi.note_number_to_name(lo)} – "
              f"{pretty_midi.note_number_to_name(hi)} ({lo}–{hi})")
        if self.top_notes:
            print(f"\n  Top {len(self.top_notes)} notes:")
            print(f"  {'Note':<6} {'Count':>6} {'Vel':>6} {'Dur(s)':>8}")
            print(f"  {'-'*30}")
            for n in self.top_notes:
                print(f"  {n.name:<6} {n.count:>6} {n.avg_velocity:>6.1f} {n.avg_duration_s:>8.3f}")
        print(f"{'='*48}\n")


def analyze(midi_path: str | Path, top_n: int = 5) -> MidiAnalysis:
    path = Path(midi_path)
    if not path.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    pm = pretty_midi.PrettyMIDI(str(path))
    mid = mido.MidiFile(str(path))

    # Tempo — use first set_tempo message, default 120 BPM
    tempo_us = 500_000
    for msg in mid:
        if msg.type == "set_tempo":
            tempo_us = msg.tempo
            break
    tempo_bpm = 60_000_000 / tempo_us

    # Time signature — use first found, default 4/4
    ts = "4/4"
    for msg in mid:
        if msg.type == "time_signature":
            ts = f"{msg.numerator}/{msg.denominator}"
            break

    # Collect all notes across all piano instruments
    all_notes: list[pretty_midi.Note] = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            all_notes.extend(instrument.notes)

    if not all_notes:
        raise ValueError("No notes found in MIDI file.")

    pitches = np.array([n.pitch for n in all_notes])
    velocities = np.array([n.velocity for n in all_notes])
    durations = np.array([n.end - n.start for n in all_notes])

    # Per-pitch statistics
    note_map: dict[int, list[pretty_midi.Note]] = {}
    for note in all_notes:
        note_map.setdefault(note.pitch, []).append(note)

    top_notes = sorted(note_map.items(), key=lambda kv: len(kv[1]), reverse=True)[:top_n]
    top_stats = [
        NoteStats(
            pitch=p,
            name=pretty_midi.note_number_to_name(p),
            count=len(notes),
            avg_velocity=float(np.mean([n.velocity for n in notes])),
            avg_duration_s=float(np.mean([n.end - n.start for n in notes])),
        )
        for p, notes in top_notes
    ]

    return MidiAnalysis(
        path=path,
        duration_s=pm.get_end_time(),
        tempo_bpm=tempo_bpm,
        time_signature=ts,
        total_notes=len(all_notes),
        unique_pitches=int(np.unique(pitches).size),
        avg_velocity=float(velocities.mean()),
        pitch_range=(int(pitches.min()), int(pitches.max())),
        top_notes=top_stats,
    )


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print("Usage: python analyze_midi.py <file.mid> [file2.mid ...]")
        print("       Drop MIDI files into the midi_files/ folder and run:")
        print("       python analyze_midi.py ../midi_files/*.mid")
        sys.exit(0)

    for path in args:
        try:
            result = analyze(path)
            result.print_report()
        except (FileNotFoundError, ValueError) as e:
            print(f"[ERROR] {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
