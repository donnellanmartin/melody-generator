import numpy as np
import sounddevice as sd
import random
from scipy.io.wavfile import write  # Import the write function to save WAV files

# Define note frequencies including sharps and flats
notes = {
    'C': 261.63,
    'C♯': 277.18,  # or D♭
    'D♭': 277.18,  # or C♯
    'D': 293.66,
    'D♯': 311.13,  # or E♭
    'E♭': 311.13,  # or D♯
    'E': 329.63,
    'F': 349.23,
    'F♯': 369.99,  # or G♭
    'G♭': 369.99,  # or F♯
    'G': 392.00,
    'G♯': 415.30,  # or A♭
    'A♭': 415.30,  # or G♯
    'A': 440.00,
    'A♯': 466.16,  # or B♭
    'B♭': 466.16,  # or A♯
    'B': 493.88
}

# Define available keys and their notes
keys = {
    "C Major": ['C', 'D', 'E', 'F', 'G', 'A', 'B'],
    "C♯ Major": ['C♯', 'D♯', 'E♯', 'F♯', 'G♯', 'A♯', 'B♯'],
    "D♭ Major": ['D♭', 'E♭', 'F', 'G♭', 'G', 'A', 'B♭'],
    "D Major": ['D', 'E', 'F♯', 'G', 'A', 'B', 'C♯'],
    "D♯ Major": ['D♯', 'E♯', 'F♯', 'G♯', 'A♯', 'B♯', 'C♯'],
    "E♭ Major": ['E♭', 'F', 'G', 'A♭', 'B♭', 'C', 'D'],
    "E Major": ['E', 'F♯', 'G♯', 'A', 'B', 'C♯', 'D♯'],
    "F Major": ['F', 'G', 'A', 'B♭', 'C', 'D', 'E'],
    "F♯ Major": ['F♯', 'G♯', 'A♯', 'B', 'C♯', 'D♯', 'E♯'],
    "G♭ Major": ['G♭', 'A♭', 'B♭', 'C♭', 'C', 'D', 'E♭'],
    "G Major": ['G', 'A', 'B', 'C', 'D', 'E', 'F♯'],
    "G♯ Major": ['G♯', 'A♯', 'B♯', 'C♯', 'D♯', 'E♯', 'F♯'],
    "A♭ Major": ['A♭', 'B♭', 'C', 'D♭', 'D', 'E', 'F'],
    "A Major": ['A', 'B', 'C♯', 'D', 'E', 'F♯', 'G♯'],
    "B♭ Major": ['B♭', 'C', 'D', 'E♭', 'F', 'G', 'A'],
    "B Major": ['B', 'C♯', 'D♯', 'E', 'F♯', 'G♯', 'A♯'],

    "A Minor": ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
    "A♯ Minor": ['A♯', 'B♯', 'C♯', 'D♯', 'E♯', 'F♯', 'G♯'],
    "B♭ Minor": ['B♭', 'C', 'D♭', 'D', 'E♭', 'F♭', 'G♭'],
    "B Minor": ['B', 'C♯', 'D', 'E', 'F♯', 'G', 'A'],
    "C Minor": ['C', 'D', 'E♭', 'F', 'G', 'A♭', 'B♭'],
    "C♯ Minor": ['C♯', 'D♯', 'E', 'F♯', 'G♯', 'A', 'B'],
    "D♭ Minor": ['D♭', 'E♭', 'F♭', 'G♭', 'A♭', 'B♭', 'C♭'],
    "D Minor": ['D', 'E', 'F', 'G', 'A', 'B♭', 'C'],
    "D♯ Minor": ['D♯', 'E♯', 'F♯', 'G♯', 'A♯', 'B', 'C♯'],
    "E♭ Minor": ['E♭', 'F', 'G♭', 'A♭', 'B♭', 'C♭', 'D♭'],
    "E Minor": ['E', 'F♯', 'G', 'A', 'B', 'C', 'D'],
    "F Minor": ['F', 'G', 'A♭', 'B♭', 'C', 'D♭', 'E♭'],
    "F♯ Minor": ['F♯', 'G♯', 'A', 'B', 'C♯', 'D', 'E'],
    "G♭ Minor": ['G♭', 'A♭', 'B♭', 'C♭', 'D♭', 'E♭', 'F♭'],
    "G Minor": ['G', 'A', 'B♭', 'C', 'D', 'E♭', 'F'],
    "G♯ Minor": ['G♯', 'A♯', 'B', 'C♯', 'D♯', 'E', 'F♯'],
    "A♭ Minor": ['A♭', 'B♭', 'C♭', 'D♭', 'E♭', 'F♭', 'G♭'],

    "C Phrygian": ['C', 'D♭', 'E♭', 'F', 'G', 'A♭', 'B♭'],
    "C♯ Phrygian": ['C♯', 'D', 'E', 'F♯', 'G♯', 'A', 'B'],
    "D♭ Phrygian": ['D♭', 'E♭', 'F', 'G♭', 'A♭', 'B♭', 'C♭'],
    "D Phrygian": ['D', 'E♭', 'F', 'G', 'A', 'B♭', 'C'],
    "D♯ Phrygian": ['D♯', 'E', 'F♯', 'G♯', 'A♯', 'B', 'C♯'],
    "E♭ Phrygian": ['E♭', 'F', 'G♭', 'A♭', 'B♭', 'C♭', 'D♭'],
    "E Phrygian": ['E', 'F', 'G', 'A', 'B', 'C', 'D'],
    "F Phrygian": ['F', 'G♭', 'A♭', 'B♭', 'C', 'D♭', 'E♭'],
    "F♯ Phrygian": ['F♯', 'G', 'A', 'B', 'C♯', 'D', 'E'],
    "G♭ Phrygian": ['G♭', 'A♭', 'B♭', 'C♭', 'D♭', 'E♭', 'F♭'],
    "G Phrygian": ['G', 'A♭', 'B♭', 'C', 'D', 'E♭', 'F'],
    "G♯ Phrygian": ['G♯', 'A', 'B', 'C♯', 'D♯', 'E', 'F♯'],
    "A♭ Phrygian": ['A♭', 'B♭', 'C♭', 'D♭', 'E♭', 'F♭', 'G♭'],
    "A Phrygian": ['A', 'B♭', 'C', 'D', 'E', 'F', 'G'],
    "B♭ Phrygian": ['B♭', 'C♭', 'D♭', 'E♭', 'F', 'G♭', 'A♭'],
    "B Phrygian": ['B', 'C', 'D', 'E', 'F♯', 'G', 'A']
}

# Prompt for BPM and harmony options
bpm = int(input("Enter BPM: "))
lower_harmony = input("Would you like a lower harmony? (Y/N): ").strip().upper() == 'Y'

# Prompt for key/mode input
key = input("Enter a key or mode (e.g., Ab Minor): ")
selected_key = keys.get(key)  # This retrieves the notes for the selected key

# Check if the selected key is valid
if not selected_key:
    print("Invalid key selected. Please restart the program and choose a valid key.")
    exit(1)

# Calculate note duration
beat_duration = 60000 / bpm  # Duration of a beat in milliseconds
note_duration = beat_duration / 4  # Quarter note duration

# Generate unique melody with 16 random notes from the selected key
melody = random.choices(selected_key, k=16)
print("Melody:", melody)

# Generate and play the song
full_song_waveform = []  # List to hold the combined waveform of the whole song
for note in melody:
    if note in notes:  # Check if the note exists in the notes dictionary
        print(f"Playing note: {note} with frequency {notes[note]}")

        # Create time array for one note
        t = np.linspace(0, note_duration / 1000, int(44100 * (note_duration / 1000)), endpoint=False)

        # Generate sine wave for the original note
        original_waveform = 0.5 * np.sin(2 * np.pi * notes[note] * t)

        # Initialize combined waveform with original
        combined_waveform = original_waveform

        # If user wants lower harmony, create lower octave sawtooth waveform
        if lower_harmony:
            lower_frequency = notes[note] / 2  # Lower octave frequency

            # Generate sawtooth wave for lower octave
            lower_waveform = 0.5 * (2 * (t * lower_frequency - np.floor(t * lower_frequency + 0.5)))

            # Combine both waveforms
            combined_waveform += lower_waveform
            print(f"Playing lower octave (saw wave): {note} with frequency {lower_frequency}")

        # Store the combined waveform for saving
        full_song_waveform.append(combined_waveform)

        # Play combined sound
        sd.play(combined_waveform, samplerate=44100)

        # Wait for sound to finish
        sd.wait()

# After playback, concatenate the waveforms
full_song_waveform = np.concatenate(full_song_waveform)

# Ask if the user wants to save the melody
save_file = input("Do you want to save the melody? (Y/N): ").strip().upper() == 'Y'
if save_file:
    filename = f"Melody: {''.join(melody)}, BPM: {bpm}, Key: {key}.wav"
    write(filename, 44100, full_song_waveform.astype(np.float32))  # Save as WAV file
    print(f"Melody saved as {filename}.")
