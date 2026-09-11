import sys
import time
import wave
import queue
from collections import deque

import numpy as np
import sounddevice as sd
import pyqtgraph as pg

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QComboBox, QCheckBox, QSlider, QSpinBox,
    QFileDialog, QMessageBox, QGroupBox, QPlainTextEdit
)


LICENSE_TEXT = """MIT License

Copyright (c) 2026 MoonCherryFox Moon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


SAMPLE_RATE = 48000
BLOCK_SIZE = 256
ANALYSIS_SIZE = 4096

EXERCISES = [
    "I speak calmly and freely. "
    "My voice sounds smooth and relaxed, without unnecessary tension.\n"
    "Сегодня я говорю спокойно и свободно. "
    "Мой голос звучит ровно, без лишнего напряжения.",

    "M-m-m… Mi, me, ma, mo, mu. "
    "Repeat softly at a comfortable pitch, do not press your voice.\n"
    "М-м-м… Ми, мэ, ма, мо, му. "
    "Повторяйте мягко, на удобной высоте, не давите на голос.",

    "The sky outside was gradually getting brighter. "
    "On the quiet street, the first passersby began to appear.\n"
    "За окном постепенно светлело. "
    "На тихой улице появлялись первые прохожие.",

    "Say the same sentence first as a question, "
    "then as a statement. Keep a comfortable volume.\n"
    "Произнесите одну и ту же фразу сначала как вопрос, "
    "затем как утверждение. Сохраняйте комфортную громкость."
]


def analyze(samples, rate):
    """Estimate F0, level, spectrum, and spectral centroid."""
    x = np.asarray(samples, dtype=np.float64)
    x = x - np.mean(x)

    rms = np.sqrt(np.mean(x * x))
    db = 20 * np.log10(max(rms, 1e-8))

    spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    frequencies = np.fft.rfftfreq(len(x), 1 / rate)

    mask = (frequencies >= 100) & (frequencies <= 5000)
    energy = spectrum[mask] ** 2
    centroid = (
        float(np.sum(frequencies[mask] * energy) / np.sum(energy))
        if np.sum(energy) > 1e-12 else 0.0
    )

    pitch = None

    if db > -55:
        # FFT autocorrelation with overlap-length correction.
        nfft = 1 << (2 * len(x) - 1).bit_length()
        fft = np.fft.rfft(x, nfft)
        ac = np.fft.irfft(fft * np.conj(fft), nfft)[:len(x)]
        ac /= np.arange(len(x), 0, -1)
        ac /= max(ac[0], 1e-12)

        lo = int(rate / 550)
        hi = min(int(rate / 60), len(ac) - 2)

        candidates = np.arange(lo + 1, hi)
        candidates = candidates[
            (ac[candidates] > ac[candidates - 1]) &
            (ac[candidates] >= ac[candidates + 1])
        ]

        if len(candidates):
            best = np.max(ac[candidates])
            if best >= 0.45:
                # The first sufficiently strong peak:
                # lowers the chance of selecting a multiple period.
                lag = int(candidates[ac[candidates] >= best * 0.90][0])
                a, b, c = ac[lag - 1:lag + 2]
                denominator = a - 2 * b + c
                offset = (
                    0.5 * (a - c) / denominator
                    if abs(denominator) > 1e-12 else 0.0
                )
                offset = float(np.clip(offset, -0.5, 0.5))
                pitch = float(rate / (lag + offset))

    return pitch, db, centroid, frequencies, spectrum


def musical_note(hz):
    midi = int(round(69 + 12 * np.log2(hz / 440)))
    names = [
        "C", "C♯", "D", "D♯", "E", "F",
        "F♯", "G", "G♯", "A", "A♯", "B"
    ]
    return f"{names[midi % 12]}{midi // 12 - 1}"


class VoiceDesktop(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Voice Desktop — voice training")
        self.resize(1100, 850)

        self.stream = None
        self.audio_queue = queue.Queue(maxsize=128)
        self.buffer = np.zeros(ANALYSIS_SIZE, dtype=np.float32)

        self.monitor_enabled = False
        self.monitor_gain = 0.5
        self.tone_enabled = False
        self.tone_frequency = 180
        self.phase = 0.0

        self.recording = False
        self.record_chunks = []
        self.recorded = np.empty(0, dtype=np.float32)
        self.playback = None
        self.play_position = 0

        self.history = deque(maxlen=250)
        self.pitch_sum = 0.0
        self.pitch_count = 0
        self.session_start = None

        self.build_ui()
        self.refresh_devices()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_audio)
        self.timer.start(40)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        devices = QHBoxLayout()
        self.mic_box = QComboBox()
        self.output_box = QComboBox()
        self.refresh_button = QPushButton("Refresh devices")
        self.start_button = QPushButton("Start")

        devices.addWidget(QLabel("Microphone:"))
        devices.addWidget(self.mic_box, 1)
        devices.addWidget(QLabel("Output:"))
        devices.addWidget(self.output_box, 1)
        devices.addWidget(self.refresh_button)
        devices.addWidget(self.start_button)
        layout.addLayout(devices)

        self.refresh_button.clicked.connect(self.refresh_devices)
        self.start_button.clicked.connect(self.toggle_stream)

        controls = QHBoxLayout()

        self.monitor_check = QCheckBox("Listen to myself")
        self.monitor_check.toggled.connect(
            lambda value: setattr(self, "monitor_enabled", value)
        )
        controls.addWidget(self.monitor_check)
        controls.addWidget(QLabel("Monitoring volume:"))

        gain = QSlider(Qt.Orientation.Horizontal)
        gain.setRange(0, 100)
        gain.setValue(50)
        gain.setMaximumWidth(160)
        gain.valueChanged.connect(
            lambda value: setattr(self, "monitor_gain", value / 100)
        )
        controls.addWidget(gain)

        self.tone_check = QCheckBox("Reference tone")
        self.tone_check.toggled.connect(
            lambda value: setattr(self, "tone_enabled", value)
        )
        controls.addWidget(self.tone_check)

        tone = QSpinBox()
        tone.setRange(60, 550)
        tone.setValue(180)
        tone.setSuffix(" Hz")
        tone.valueChanged.connect(
            lambda value: setattr(self, "tone_frequency", value)
        )
        controls.addWidget(tone)
        controls.addStretch()
        layout.addLayout(controls)

        warning = QLabel(
            "Use headphones: playback through speakers may cause "
            "whistling and distort measurements."
        )
        warning.setStyleSheet("color: #e9b66b;")
        layout.addWidget(warning)

        readings = QGridLayout()
        self.pitch_label = QLabel("— Hz")
        self.pitch_label.setStyleSheet(
            "font-size: 40px; font-weight: bold; color: #67d9ef;"
        )
        self.note_label = QLabel("Note: —")
        self.db_label = QLabel("Level: — dBFS")
        self.centroid_label = QLabel("Spectral centroid: —")
        self.average_label = QLabel("Average frequency: —")
        self.time_label = QLabel("Session: 00:00")

        readings.addWidget(self.pitch_label, 0, 0)
        readings.addWidget(self.note_label, 1, 0)
        readings.addWidget(self.db_label, 0, 1)
        readings.addWidget(self.centroid_label, 1, 1)
        readings.addWidget(self.average_label, 0, 2)
        readings.addWidget(self.time_label, 1, 2)
        layout.addLayout(readings)

        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Target range:"))

        self.target_low = QSpinBox()
        self.target_high = QSpinBox()
        for spin in (self.target_low, self.target_high):
            spin.setRange(60, 550)
            spin.setSuffix(" Hz")

        self.target_low.setValue(140)
        self.target_high.setValue(220)

        target_row.addWidget(self.target_low)
        target_row.addWidget(QLabel("—"))
        target_row.addWidget(self.target_high)

        reset = QPushButton("Reset statistics")
        reset.clicked.connect(self.reset_statistics)
        target_row.addWidget(reset)
        target_row.addStretch()
        layout.addLayout(target_row)

        self.pitch_plot = pg.PlotWidget(title="Fundamental frequency")
        self.pitch_plot.setLabel("left", "F0", units="Hz")
        self.pitch_plot.setLabel("bottom", "Recent updates")
        self.pitch_plot.setYRange(60, 550)
        self.pitch_plot.setXRange(0, 249)
        self.pitch_plot.showGrid(x=True, y=True, alpha=0.2)

        self.target_region = pg.LinearRegionItem(
            values=(140, 220),
            orientation="horizontal",
            movable=False,
            brush=pg.mkBrush(70, 180, 110, 35)
        )
        self.target_region.setZValue(-10)
        self.pitch_plot.addItem(self.target_region)
        self.pitch_curve = self.pitch_plot.plot(
            pen=pg.mkPen("#67d9ef", width=2),
            connect="finite"
        )
        layout.addWidget(self.pitch_plot, 2)

        self.spectrum_plot = pg.PlotWidget(
            title="Spectrum — timbre guide, not a precise resonance measurement"
        )
        self.spectrum_plot.setLabel("bottom", "Frequency", units="Hz")
        self.spectrum_plot.setLabel("left", "Relative level", units="dB")
        self.spectrum_plot.setXRange(0, 5000)
        self.spectrum_plot.setYRange(-80, 0)
        self.spectrum_plot.showGrid(x=True, y=True, alpha=0.2)
        self.spectrum_curve = self.spectrum_plot.plot(
            pen=pg.mkPen("#cf9fff", width=1.5)
        )
        layout.addWidget(self.spectrum_plot, 1)

        recording_row = QHBoxLayout()
        self.record_button = QPushButton("● Record")
        self.play_button = QPushButton("▶ Play recording")
        self.save_button = QPushButton("Save WAV")

        self.record_button.clicked.connect(self.toggle_recording)
        self.play_button.clicked.connect(self.play_recording)
        self.save_button.clicked.connect(self.save_recording)

        recording_row.addWidget(self.record_button)
        recording_row.addWidget(self.play_button)
        recording_row.addWidget(self.save_button)
        recording_row.addStretch()
        layout.addLayout(recording_row)

        exercises = QGroupBox("Exercises — without tension or pain")
        exercise_layout = QVBoxLayout(exercises)
        exercise_selector = QComboBox()
        exercise_selector.addItems([
            "Calm speech",
            "Soft syllables",
            "Reading",
            "Intonation"
        ])

        self.exercise_text = QPlainTextEdit(EXERCISES[0])
        self.exercise_text.setMaximumHeight(85)
        exercise_selector.currentIndexChanged.connect(
            lambda index: self.exercise_text.setPlainText(EXERCISES[index])
        )
        exercise_layout.addWidget(exercise_selector)
        exercise_layout.addWidget(self.exercise_text)
        layout.addWidget(exercises)

        self.status_label = QLabel("Ready. Select devices and press Start.")
        layout.addWidget(self.status_label)

    def refresh_devices(self):
        if self.stream is not None:
            return

        self.mic_box.clear()
        self.output_box.clear()

        try:
            devices = sd.query_devices()
            apis = sd.query_hostapis()
            default_input, default_output = sd.default.device

            for index, device in enumerate(devices):
                api = apis[device["hostapi"]]["name"]
                name = f'{device["name"]} [{api}]'

                if device["max_input_channels"] > 0:
                    self.mic_box.addItem(name, index)
                    if index == default_input:
                        self.mic_box.setCurrentIndex(self.mic_box.count() - 1)

                if device["max_output_channels"] > 0:
                    self.output_box.addItem(name, index)
                    if index == default_output:
                        self.output_box.setCurrentIndex(
                            self.output_box.count() - 1
                        )
        except Exception as error:
            QMessageBox.warning(self, "Audio devices", str(error))

    def audio_callback(self, indata, outdata, frames, timing, status):
        # There is no F0 calculation or UI update here.
        # This reduces the risk of monitoring interruptions.
        outdata.fill(0)

        if self.monitor_enabled:
            outdata[:, 0] = indata[:, 0] * self.monitor_gain

        if self.tone_enabled:
            step = 2 * np.pi * self.tone_frequency / SAMPLE_RATE
            angles = self.phase + step * np.arange(frames)
            outdata[:, 0] += (0.08 * np.sin(angles)).astype(np.float32)
            self.phase = float((self.phase + step * frames) % (2 * np.pi))

        playback = self.playback
        if playback is not None:
            position = self.play_position
            count = min(frames, len(playback) - position)

            if count > 0:
                outdata[:count, 0] += playback[position:position + count]
                self.play_position += count

            if self.play_position >= len(playback):
                self.playback = None

        np.clip(outdata, -1, 1, out=outdata)

        try:
            self.audio_queue.put_nowait(
                (indata[:, 0].copy(), bool(status))
            )
        except queue.Full:
            pass

    def toggle_stream(self):
        if self.stream is not None:
            self.stop_stream()
            return

        if self.mic_box.currentData() is None or self.output_box.currentData() is None:
            QMessageBox.warning(self, "No device", "Select input and output.")
            return

        candidate = None
        try:
            candidate = sd.Stream(
                device=(
                    self.mic_box.currentData(),
                    self.output_box.currentData()
                ),
                samplerate=SAMPLE_RATE,
                blocksize=BLOCK_SIZE,
                channels=(1, 1),
                dtype="float32",
                latency="low",
                callback=self.audio_callback
            )
            candidate.start()
            self.stream = candidate
        except Exception as error:
            if candidate is not None:
                candidate.close()
            QMessageBox.critical(
                self,
                "Failed to start audio",
                f"{error}\n\n"
                "Try using the same API for both input and output, for example WASAPI. "
                "Devices must support 48 kHz."
            )
            return

        self.buffer.fill(0)
        self.reset_statistics()
        self.start_button.setText("Stop")
        self.mic_box.setEnabled(False)
        self.output_box.setEnabled(False)
        self.refresh_button.setEnabled(False)

        latency_in, latency_out = self.stream.latency
        self.status_label.setText(
            "Running. Driver-reported input + output latency: "
            f"{1000 * (latency_in + latency_out):.0f} ms. "
            "F0 analysis window: 85 ms."
        )

    def stop_stream(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        # Drain any remaining blocks before finishing the recording.
        self.update_audio()

        if self.recording:
            self.finish_recording()

        self.playback = None
        self.start_button.setText("Start")
        self.mic_box.setEnabled(True)
        self.output_box.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.status_label.setText("Stopped.")

    def reset_statistics(self):
        self.history.clear()
        self.pitch_curve.setData([])
        self.pitch_sum = 0
        self.pitch_count = 0
        self.average_label.setText("Average frequency: —")
        self.session_start = time.monotonic()
        self.time_label.setText("Session: 00:00")

    def update_audio(self):
        chunks = []
        had_audio_issue = False

        while True:
            try:
                chunk, issue = self.audio_queue.get_nowait()
            except queue.Empty:
                break

            chunks.append(chunk)
            had_audio_issue |= issue

            if self.recording:
                self.record_chunks.append(chunk)

        low = min(self.target_low.value(), self.target_high.value())
        high = max(self.target_low.value(), self.target_high.value())
        self.target_region.setRegion((low, high))

        if self.stream is not None and self.session_start is not None:
            elapsed = int(time.monotonic() - self.session_start)
            self.time_label.setText(
                f"Session: {elapsed // 60:02}:{elapsed % 60:02}"
            )

        if not chunks:
            return

        new_audio = np.concatenate(chunks)
        if len(new_audio) >= ANALYSIS_SIZE:
            self.buffer[:] = new_audio[-ANALYSIS_SIZE:]
        else:
            n = len(new_audio)
            self.buffer[:-n] = self.buffer[n:]
            self.buffer[-n:] = new_audio

        pitch, db, centroid, frequencies, spectrum = analyze(
            self.buffer, SAMPLE_RATE
        )

        self.db_label.setText(f"Level: {db:.1f} dBFS")

        if db > -55:
            self.centroid_label.setText(
                f"Spectral centroid: {centroid:.0f} Hz"
            )
        else:
            self.centroid_label.setText("Spectral centroid: —")

        if pitch is not None:
            self.pitch_label.setText(f"{pitch:.1f} Hz")
            self.note_label.setText(f"Note: {musical_note(pitch)}")

            color = "#7fe0a0" if low <= pitch <= high else "#efc16f"
            self.pitch_label.setStyleSheet(
                f"font-size: 40px; font-weight: bold; color: {color};"
            )

            self.pitch_sum += pitch
            self.pitch_count += 1
            self.average_label.setText(
                "Average frequency: "
                f"{self.pitch_sum / self.pitch_count:.1f} Hz"
            )
            self.history.append(pitch)
        else:
            self.pitch_label.setText("— Hz")
            self.note_label.setText("Note: no stable tone")
            self.history.append(float("nan"))

        self.pitch_curve.setData(
            np.arange(len(self.history)),
            np.asarray(self.history),
            connect="finite"
        )

        if db > -65:
            relative = 20 * np.log10(
                np.maximum(spectrum, 1e-12) /
                max(float(np.max(spectrum)), 1e-12)
            )
            self.spectrum_curve.setData(
                frequencies, np.maximum(relative, -80)
            )
        else:
            self.spectrum_curve.setData([], [])

        if had_audio_issue:
            self.status_label.setText(
                "Audio drop detected. Try BLOCK_SIZE = 512 "
                "or another audio device."
            )

    def toggle_recording(self):
        if self.recording:
            self.update_audio()
            self.finish_recording()
            return

        if self.stream is None:
            QMessageBox.information(
                self, "Recording", "Start the microphone first."
            )
            return

        self.update_audio()
        self.playback = None
        self.record_chunks = []
        self.recording = True
        self.record_button.setText("■ Finish recording")
        self.play_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def finish_recording(self):
        self.recording = False
        self.recorded = (
            np.concatenate(self.record_chunks)
            if self.record_chunks else np.empty(0, dtype=np.float32)
        )
        self.record_chunks = []
        self.record_button.setText("● Record")
        self.play_button.setEnabled(True)
        self.save_button.setEnabled(True)
        self.status_label.setText(
            f"Recorded: {len(self.recorded) / SAMPLE_RATE:.1f} sec."
        )

    def play_recording(self):
        if self.stream is None:
            QMessageBox.information(
                self, "Playback", "Press Start first."
            )
            return

        if not len(self.recorded):
            QMessageBox.information(
                self, "Playback", "Record your voice first."
            )
            return

        if self.playback is not None:
            return

        # Do not mix the recording with the microphone monitoring and reference tone.
        self.monitor_check.setChecked(False)
        self.tone_check.setChecked(False)
        self.play_position = 0
        self.playback = self.recorded

    def save_recording(self):
        if not len(self.recorded):
            QMessageBox.information(
                self, "Saving", "No recording available to save."
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save recording", "voice.wav", "WAV (*.wav)"
        )
        if not path:
            return
        if not path.lower().endswith(".wav"):
            path += ".wav"

        try:
            pcm = (
                np.clip(self.recorded, -1, 1) * 32767
            ).astype("<i2")

            with wave.open(path, "wb") as file:
                file.setnchannels(1)
                file.setsampwidth(2)
                file.setframerate(SAMPLE_RATE)
                file.writeframes(pcm.tobytes())

            self.status_label.setText(f"Saved: {path}")
        except Exception as error:
            QMessageBox.critical(self, "Save error", str(error))

    def closeEvent(self, event):
        self.timer.stop()
        self.stop_stream()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet("""
        QWidget {
            background: #191d26;
            color: #e5eaf2;
            font-size: 13px;
        }
        QPushButton, QComboBox, QSpinBox {
            background: #2a3241;
            border: 1px solid #45516a;
            border-radius: 5px;
            padding: 6px;
        }
        QPushButton:hover { background: #37455e; }
        QPushButton:disabled { color: #77808e; }
        QPlainTextEdit {
            background: #222936;
            border: 1px solid #45516a;
        }
        QGroupBox {
            border: 1px solid #45516a;
            margin-top: 10px;
            padding-top: 12px;
        }
    """)

    pg.setConfigOptions(
        antialias=True,
        background="#151922",
        foreground="#cdd5e3"
    )

    window = VoiceDesktop()
    window.show()
    sys.exit(app.exec())
