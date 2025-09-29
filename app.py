import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging

# Try to import TensorFlow and handle version issues
try:
    import tensorflow as tf
    from tensorflow import keras

    print(f"TensorFlow version: {tf.__version__}")

    # For TensorFlow 2.15+ compatibility
    if hasattr(tf.config, 'experimental'):
        try:
            tf.config.experimental.enable_memory_growth = True
        except:
            pass

except ImportError as e:
    print(f"TensorFlow import error: {e}")
    # Fallback imports or error handling
    pass
import numpy as np
import librosa
import pickle
import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
import logging
from scipy.signal import butter, lfilter, find_peaks, welch
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import scipy.signal as signal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

model = None
extractor = None
label_encoder = None
metadata = None



def convert_to_serializable(obj):
    """Fixed version that properly handles Python booleans"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, bool):  # Add explicit Python bool handling
        return obj  # Keep as-is, don't convert
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class MultiDurationAngklungExtractor:
    """Enhanced extractor that handles both long and short notes"""

    def __init__(self, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.scaler = StandardScaler()
        self.scaler_fitted = False
        self.target_frames = 87

        self.note_frequencies = {
            'do': 261.63, 're': 293.66, 'mi': 329.63, 'fa': 349.23,
            'sol': 392.00, 'la': 440.00, 'ti': 493.88, 'do_high': 523.25
        }

        self.note_ranges = {}
        for note, freq in self.note_frequencies.items():
            # FIXED: Much tighter tolerances to reduce confusion
            if note == 're':
                tolerance = 0.04  # Tightest for RE (was 0.06)
            elif note == 'do':
                tolerance = 0.05  # Tighter for DO to avoid TI confusion (was 0.08)
            elif note == 'ti':
                tolerance = 0.05  # Much tighter for TI to avoid DO confusion (was 0.10)
            elif note == 'fa':
                tolerance = 0.06  # Tighter for FA to avoid MI confusion (was 0.10)
            elif note == 'mi':
                tolerance = 0.06  # Tighter for MI to avoid FA confusion (was 0.12)
            elif note == 'do_high':
                tolerance = 0.05  # Tighter for DO_HIGH (was 0.07)
            elif note == 'sol':
                tolerance = 0.08  # Slightly tighter (was 0.12)
            elif note == 'la':
                tolerance = 0.09  # Slightly tighter (was 0.09)
            else:
                tolerance = 0.08

            self.note_ranges[note] = {
                'center': freq,
                'min': freq * (1 - tolerance),
                'max': freq * (1 + tolerance),
                'harmonics': [freq * h for h in [1, 2, 3, 4, 5]]
            }

    def detect_note_duration_type(self, y, sr):
        """Detect if this is a short or long note"""
        # Remove silence from beginning and end
        non_silent = np.abs(y) > 0.01 * np.max(np.abs(y))
        if np.any(non_silent):
            start_idx = np.where(non_silent)[0][0]
            end_idx = np.where(non_silent)[0][-1]
            effective_duration = (end_idx - start_idx) / sr
        else:
            effective_duration = len(y) / sr

        # Classify as short if less than 1.5 seconds of actual sound
        return 'short' if effective_duration < 1.5 else 'long'

    def advanced_distance_normalization(self, y, duration_type='long'):
        """Enhanced normalization that considers note duration"""
        if len(y) == 0:
            return y

        y = y - np.mean(y)
        max_amp = np.max(np.abs(y))
        rms_amp = np.sqrt(np.mean(y ** 2))

        # Different normalization strategies for short vs long notes
        if duration_type == 'short':
            # More conservative normalization for short notes to avoid over-amplification
            if max_amp < 0.001:  # Very quiet - likely silence
                return y * 0  # Return silence
            elif max_amp < 0.01:
                y = y * 5000  # Less aggressive amplification
                y = np.sign(y) * np.power(np.abs(y), 0.8)  # Less compression
            elif max_amp < 0.1:
                y = y * 2000
                y = np.sign(y) * np.power(np.abs(y), 0.75)
            else:
                # Good signal level - minimal processing
                if rms_amp > 0:
                    target_rms = 0.12  # Lower target for short notes
                    y = y / rms_amp * target_rms
        else:
            # Long notes: original normalization
            if max_amp < 0.005:
                y = y * 50000
                y = np.sign(y) * np.power(np.abs(y), 0.85)
            elif max_amp < 0.05:
                y = y * 5000
                y = np.sign(y) * np.power(np.abs(y), 0.8)
            elif max_amp > 0.7:
                y = y * 0.3
                y = np.sign(y) * np.power(np.abs(y), 0.5)
            elif max_amp > 0.3:
                y = y * 0.6
                y = np.sign(y) * np.power(np.abs(y), 0.7)
            else:
                if rms_amp > 0:
                    target_rms = 0.1
                    y = y / rms_amp * target_rms

        max_final = np.max(np.abs(y))
        if max_final > 0.95:
            y = y * (0.95 / max_final)

        return y

    def extract_fundamental_with_duration_awareness(self, y, sr, duration_type='long'):
        """Enhanced fundamental extraction that considers note duration"""
        try:
            y_normalized = self.advanced_distance_normalization(y.copy(), duration_type)
            detected_frequencies = []

            if np.max(np.abs(y_normalized)) < 0.01:
                # FIXED: Don't return 0 for short notes with valid signals
                if duration_type == 'short' and np.max(np.abs(y)) > 0.001:
                    # Continue with frequency extraction for short notes
                    pass
                elif np.max(np.abs(y)) < 0.0001:
                    return 0.0  # Only return 0 for truly silent recordings

            detected_frequencies = []

            fft = np.abs(np.fft.fft(y_normalized))
            freqs = np.fft.fftfreq(len(y_normalized), 1 / sr)
            valid_idx = (freqs >= 220) & (freqs <= 580) & (freqs > 0)

            if np.any(valid_idx):
                valid_fft = fft[valid_idx]
                valid_freqs = freqs[valid_idx]

                max_amp = np.max(np.abs(y))

                # Adjust prominence based on duration type
                if duration_type == 'short':
                    if max_amp < 0.002:  # More lenient (was 0.005)
                        prominence_factor = 0.03  # Lower threshold (was 0.05)
                    elif max_amp < 0.01:
                        prominence_factor = 0.04  # Lower threshold (was 0.05)
                    else:
                        prominence_factor = 0.06  # Lower threshold (was 0.08)
                else:
                    # Keep original values for long notes
                    if max_amp < 0.01:
                        prominence_factor = 0.03
                    elif max_amp > 0.5:
                        prominence_factor = 0.15
                    else:
                        prominence_factor = 0.08

                prominence = np.max(valid_fft) * prominence_factor
                peaks, _ = find_peaks(valid_fft, prominence=prominence, distance=5)

                if len(peaks) > 0:
                    peak_freqs = valid_freqs[peaks]
                    peak_powers = valid_fft[peaks]
                    strongest_idx = np.argmax(peak_powers)
                    fundamental_candidate = float(peak_freqs[strongest_idx])
                    detected_frequencies.append(fundamental_candidate)

            # For short notes, use a shorter analysis window for autocorr
            try:
                if duration_type == 'short':
                    window_size = min(len(y_normalized), sr // 2)  # Shorter window
                else:
                    window_size = min(len(y_normalized), sr)

                start_idx = len(y_normalized) // 4
                end_idx = start_idx + window_size
                y_windowed = y_normalized[start_idx:min(end_idx, len(y_normalized))]

                if len(y_windowed) > 500:  # Reduced minimum length for short notes
                    autocorr = np.correlate(y_windowed, y_windowed, mode='full')
                    autocorr = autocorr[autocorr.size // 2:]

                    min_period = int(sr / 580)
                    max_period = int(sr / 220)
                    search_range = autocorr[min_period:min_period + max_period]

                    if len(search_range) > 0:
                        peak_idx = np.argmax(search_range) + min_period
                        if peak_idx > 0:
                            pitch_freq = sr / peak_idx
                            if 220 <= pitch_freq <= 580:
                                detected_frequencies.append(float(pitch_freq))
            except:
                pass

            # Adjusted piptrack for short notes
            try:
                hop_length_pitch = 128 if duration_type == 'short' else 256  # Smaller hop for short notes
                pitches, magnitudes = librosa.core.piptrack(
                    y=y_normalized, sr=sr, fmin=220, fmax=580, hop_length=hop_length_pitch
                )

                pitch_candidates = []
                for t in range(pitches.shape[1]):
                    mag_col = magnitudes[:, t]
                    if np.max(mag_col) > 0:
                        max_idx = np.argmax(mag_col)
                        pitch = pitches[max_idx, t]
                        if pitch > 0 and 220 <= pitch <= 580:
                            pitch_candidates.append(pitch)

                min_candidates = 2 if duration_type == 'short' else 3  # Lower threshold for short notes
                if len(pitch_candidates) >= min_candidates:
                    sorted_candidates = np.sort(pitch_candidates)
                    q1_idx = len(sorted_candidates) // 4
                    q3_idx = 3 * len(sorted_candidates) // 4
                    median_pitch = np.median(sorted_candidates[q1_idx:q3_idx])
                    detected_frequencies.append(float(median_pitch))
            except:
                pass

            if not detected_frequencies:
                best_freq = None
                best_score = 0

                for note, expected_freq in self.note_frequencies.items():
                    if note == 're':
                        tolerance = 15
                    else:
                        tolerance = 25 if note in ['do', 'do_high', 'ti'] else 30

                    freq_mask = (freqs >= expected_freq - tolerance) & \
                                (freqs <= expected_freq + tolerance)

                    if np.any(freq_mask):
                        energy_around_note = np.sum(fft[freq_mask])
                        if energy_around_note > best_score:
                            best_score = energy_around_note
                            best_freq = expected_freq

                if best_freq:
                    detected_frequencies.append(best_freq)

            if detected_frequencies:
                if len(detected_frequencies) == 1:
                    return detected_frequencies[0]
                else:
                    detected_frequencies = np.array(detected_frequencies)
                    median_freq = np.median(detected_frequencies)
                    deviations = np.abs(detected_frequencies - median_freq)

                    # More lenient for short notes
                    max_deviation = 60 if duration_type == 'short' else 50
                    valid_frequencies = detected_frequencies[deviations <= max_deviation]

                    if len(valid_frequencies) > 0:
                        return float(np.mean(valid_frequencies))
                    else:
                        return float(median_freq)

            return 350.0

        except Exception as e:
            return 350.0

    def _extract_short_duration_discriminative_features(self, y, sr, fundamental_freq):
        """Extract features specifically for short duration note discrimination"""
        fft = np.abs(np.fft.fft(y))
        freqs = np.fft.fftfreq(len(y), 1 / sr)

        # SOL vs LA discrimination (392 Hz vs 440 Hz)
        sol_freq = 392.0
        la_freq = 440.0
        sol_energy = self._get_energy_around_freq(fft, freqs, sol_freq, 15)
        la_energy = self._get_energy_around_freq(fft, freqs, la_freq, 15)

        # FA vs MI discrimination (349 Hz vs 330 Hz)
        fa_freq = 349.23
        mi_freq = 329.63
        fa_energy = self._get_energy_around_freq(fft, freqs, fa_freq, 12)
        mi_energy = self._get_energy_around_freq(fft, freqs, mi_freq, 12)

        # Frequency precision scoring
        sol_precision = 1.0 / (1.0 + abs(fundamental_freq - sol_freq) / 15)
        la_precision = 1.0 / (1.0 + abs(fundamental_freq - la_freq) / 15)
        fa_precision = 1.0 / (1.0 + abs(fundamental_freq - fa_freq) / 10)
        mi_precision = 1.0 / (1.0 + abs(fundamental_freq - mi_freq) / 10)

        return {
            'sol_la_energy_ratio': sol_energy / (la_energy + 1e-8),
            'fa_mi_energy_ratio': fa_energy / (mi_energy + 1e-8),
            'sol_precision_score': sol_precision,
            'la_precision_score': la_precision,
            'fa_precision_score': fa_precision,
            'mi_precision_score': mi_precision,
            'frequency_specificity': min(sol_precision - la_precision, fa_precision - mi_precision)
        }

    def extract_duration_specific_features(self, y, sr, fundamental_freq, duration_type):
        """Extract features specific to note duration"""
        features = {}

        try:
            # 1. Duration characteristics
            non_silent = np.abs(y) > 0.01 * np.max(np.abs(y))
            if np.any(non_silent):
                start_idx = np.where(non_silent)[0][0]
                end_idx = np.where(non_silent)[0][-1]
                effective_duration = (end_idx - start_idx) / sr
                onset_time = start_idx / sr
                sustain_ratio = effective_duration / (len(y) / sr)
            else:
                effective_duration = len(y) / sr
                onset_time = 0
                sustain_ratio = 1.0

            # 2. Attack characteristics (important for short notes)
            if len(y) > sr // 10:  # At least 0.1 seconds
                attack_window = y[:sr // 10]  # First 0.1 seconds
                attack_energy = np.sum(attack_window ** 2)
                total_energy = np.sum(y ** 2)
                attack_ratio = attack_energy / (total_energy + 1e-8)

                # Peak attack time
                peak_idx = np.argmax(np.abs(attack_window))
                peak_attack_time = peak_idx / sr
            else:
                attack_ratio = 0.5
                peak_attack_time = 0.01

            # 3. Decay characteristics
            if len(y) > sr // 5:  # At least 0.2 seconds
                decay_window = y[-sr // 5:]  # Last 0.2 seconds
                decay_energy = np.sum(decay_window ** 2)
                decay_ratio = decay_energy / (total_energy + 1e-8)

                # Decay slope
                abs_decay = np.abs(decay_window)
                if len(abs_decay) > 1:
                    decay_slope = (abs_decay[-1] - abs_decay[0]) / len(abs_decay)
                else:
                    decay_slope = 0
            else:
                decay_ratio = 0.3
                decay_slope = -0.1

            # 4. Harmonic development (different for short vs long)
            fft = np.abs(np.fft.fft(y))
            freqs = np.fft.fftfreq(len(y), 1 / sr)

            # Early vs late harmonic content (for long notes)
            if duration_type == 'long' and len(y) > sr:
                early_y = y[:sr // 2]  # First half second
                late_y = y[sr // 2:]  # Rest of the signal

                early_fft = np.abs(np.fft.fft(early_y))
                late_fft = np.abs(np.fft.fft(late_y[:len(early_y)]))  # Same length

                early_freqs = np.fft.fftfreq(len(early_y), 1 / sr)

                # Harmonic energy in each period
                harmonic_freqs = [fundamental_freq * h for h in [2, 3, 4]]
                early_harmonic_energy = sum([self._get_energy_around_freq(early_fft, early_freqs, hf, 20)
                                             for hf in harmonic_freqs if hf < sr / 2])
                late_harmonic_energy = sum([self._get_energy_around_freq(late_fft, early_freqs, hf, 20)
                                            for hf in harmonic_freqs if hf < sr / 2])

                harmonic_development = late_harmonic_energy / (early_harmonic_energy + 1e-8)
            else:
                harmonic_development = 1.0

            # 5. Spectral flux (rate of spectral change)
            if len(y) > 1024:
                hop_length = 512
                stft = np.abs(librosa.stft(y, hop_length=hop_length))
                spectral_flux = np.sum(np.diff(stft, axis=1) ** 2, axis=0)
                flux_mean = np.mean(spectral_flux)
                flux_std = np.std(spectral_flux)
            else:
                flux_mean = 0.1
                flux_std = 0.05

            features.update({
                'effective_duration': effective_duration,
                'onset_time': onset_time,
                'sustain_ratio': sustain_ratio,
                'attack_ratio': attack_ratio,
                'peak_attack_time': peak_attack_time,
                'decay_ratio': decay_ratio,
                'decay_slope': decay_slope,
                'harmonic_development': harmonic_development,
                'spectral_flux_mean': flux_mean,
                'spectral_flux_std': flux_std,
                'duration_type_numeric': 0.0 if duration_type == 'short' else 1.0
            })

        except Exception as e:
            # Default values for duration features
            default_duration_features = {
                'effective_duration': 1.0,
                'onset_time': 0.05,
                'sustain_ratio': 0.8,
                'attack_ratio': 0.3,
                'peak_attack_time': 0.02,
                'decay_ratio': 0.2,
                'decay_slope': -0.1,
                'harmonic_development': 1.0,
                'spectral_flux_mean': 0.1,
                'spectral_flux_std': 0.05,
                'duration_type_numeric': 0.5
            }
            features.update(default_duration_features)

        return features

    def analyze_note_specific_features(self, y, sr, fundamental_freq, duration_type='long'):
        """Enhanced note analysis with duration awareness"""
        features = {}

        try:
            fft = np.abs(np.fft.fft(y))
            freqs = np.fft.fftfreq(len(y), 1 / sr)
            ti_freq = self.note_frequencies['ti']
            do_freq = self.note_frequencies['do']
            fa_freq = self.note_frequencies['fa']
            mi_freq = self.note_frequencies['mi']
            re_freq = self.note_frequencies['re']
            do_freq = self.note_frequencies['do']
            do_high_freq = self.note_frequencies['do_high']
            do_do_high_freq_gap = abs(do_freq - do_high_freq)  # About 261 Hz gap
            measured_do_gap = abs(fundamental_freq - do_freq)
            measured_do_high_gap = abs(fundamental_freq - do_high_freq)
            # Adjust tolerance based on duration type
            base_tolerance = 12 if duration_type == 'short' else 15  # Reduced
            re_tolerance = base_tolerance * 0.75  # RE needs tighter tolerance
            do_tolerance = base_tolerance
            ti_do_freq_gap = abs(ti_freq - do_freq)  # About 32 Hz
            measured_ti_do_position = abs(fundamental_freq - ((ti_freq + do_freq) / 2))
            ti_do_positioning_accuracy = 1.0 / (1.0 + measured_ti_do_position / ti_do_freq_gap)
            re_energy = self._get_energy_around_freq(fft, freqs, re_freq, re_tolerance)
            do_energy = self._get_energy_around_freq(fft, freqs, do_freq, do_tolerance)
            do_high_energy = self._get_energy_around_freq(fft, freqs, do_high_freq, do_tolerance)
            high_freq_threshold = 500
            high_freq_energy = np.sum(fft[freqs > high_freq_threshold])
            total_energy = np.sum(fft[freqs > 0])
            ti_high_freq_indicator = high_freq_energy / (total_energy + 1e-8)
            re_distance = abs(fundamental_freq - re_freq)
            do_distance = abs(fundamental_freq - do_freq)
            do_high_distance = abs(fundamental_freq - do_high_freq)
            fa_mi_freq_gap = abs(fa_freq - mi_freq)  # About 20 Hz
            measured_fa_mi_position = abs(fundamental_freq - ((fa_freq + mi_freq) / 2))
            fa_mi_positioning_accuracy = 1.0 / (1.0 + measured_fa_mi_position / fa_mi_freq_gap)

            # FA and MI have different harmonic patterns
            fa_2nd_harmonic = self._get_energy_around_freq(fft, freqs, fa_freq * 2, 20)
            mi_2nd_harmonic = self._get_energy_around_freq(fft, freqs, mi_freq * 2, 20)
            fa_energy = self._get_energy_around_freq(fft, freqs, fa_freq, base_tolerance)
            mi_energy = self._get_energy_around_freq(fft, freqs, mi_freq, base_tolerance)

            fa_harmonic_strength = fa_2nd_harmonic / (fa_energy + 1e-8)
            mi_harmonic_strength = mi_2nd_harmonic / (mi_energy + 1e-8)
            fa_mi_harmonic_distinction = fa_harmonic_strength / (mi_harmonic_strength + 1e-8)
            # Duration-adjusted positioning scores
            distance_factor = 1.2 if duration_type == 'short' else 1.0
            re_positioning_score = 1.0 / (1.0 + abs(fundamental_freq - re_freq) / (20 * distance_factor))
            do_positioning_score = 1.0 / (1.0 + abs(fundamental_freq - do_freq) / (20 * distance_factor))
            do_high_positioning_score = 1.0 / (1.0 + abs(fundamental_freq - do_high_freq) / (20 * distance_factor))

            # Harmonic analysis with duration consideration
            harmonic_tolerance = 25 if duration_type == 'short' else 30
            re_harmonic2 = self._get_energy_around_freq(fft, freqs, re_freq * 2, harmonic_tolerance)
            re_harmonic3 = self._get_energy_around_freq(fft, freqs, re_freq * 3, harmonic_tolerance)
            do_harmonic2 = self._get_energy_around_freq(fft, freqs, do_freq * 2, harmonic_tolerance)
            do_high_harmonic2 = self._get_energy_around_freq(fft, freqs, do_high_freq * 2, harmonic_tolerance)

            re_harmonic_ratio = re_harmonic2 / (re_harmonic3 + 1e-8)

            freq_isolation = min(abs(re_freq - do_freq), abs(re_freq - do_high_freq)) / max(abs(re_freq - do_freq),
                                                                                            abs(re_freq - do_high_freq))

            # Peak sharpness analysis
            peak_tolerance = 25 if duration_type == 'short' else 30
            re_region_mask = (freqs >= re_freq - peak_tolerance) & (freqs <= re_freq + peak_tolerance)
            if np.any(re_region_mask):
                re_region_fft = fft[re_region_mask]
                re_peak_sharpness = np.max(re_region_fft) / (np.mean(re_region_fft) + 1e-8)
            else:
                re_peak_sharpness = 1.0
            if 250 < fundamental_freq < 550:
                # Use harmonic analysis for discrimination
                do_harmonic_pattern = do_harmonic2 / (do_energy + 1e-8)
                do_high_harmonic_pattern = do_high_harmonic2 / (do_high_energy + 1e-8)

                # DO tends to have stronger fundamental, DO_HIGH has more harmonics
                if fundamental_freq < 350:  # Closer to DO range
                    do_bias = 1.5  # Bias towards DO
                    do_high_bias = 0.7
                else:  # Closer to DO_HIGH range
                    do_bias = 0.7
                    do_high_bias = 1.5

                features.update({
                    'do_enhanced_score': (do_energy * do_bias) / (measured_do_gap + 1e-8),
                    'do_high_enhanced_score': (do_high_energy * do_high_bias) / (measured_do_high_gap + 1e-8),
                })
            # Spectral centroid analysis
            centroid_range = 50 if duration_type == 'short' else 70
            re_spectral_region = (freqs >= re_freq - centroid_range) & (freqs <= re_freq + centroid_range)
            if np.any(re_spectral_region):
                re_region_centroid = np.sum(freqs[re_spectral_region] * fft[re_spectral_region]) / (
                        np.sum(fft[re_spectral_region]) + 1e-8)
                re_centroid_match = 1.0 / (1.0 + abs(re_region_centroid - re_freq) / 30)
            else:
                re_centroid_match = 0.5

            total_energy = np.sum(fft[freqs > 0])
            re_fundamental_strength = re_energy / (total_energy + 1e-8)

            features.update({
                're_energy': re_energy,
                're_do_energy_ratio': re_energy / (do_energy + 1e-8),
                're_do_high_energy_ratio': re_energy / (do_high_energy + 1e-8),
                're_distance': re_distance,
                'do_distance': do_distance,
                'do_high_distance': do_high_distance,
                're_positioning_score': re_positioning_score,
                'do_positioning_score': do_positioning_score,
                'do_high_positioning_score': do_high_positioning_score,
                're_freq_isolation': freq_isolation,
                're_harmonic_ratio': re_harmonic_ratio,
                're_harmonic2_energy': re_harmonic2,
                're_harmonic3_energy': re_harmonic3,
                're_peak_sharpness': re_peak_sharpness,
                're_centroid_match': re_centroid_match,
                're_fundamental_strength': re_fundamental_strength,
                're_distance_confidence': 1.0 / (1.0 + re_distance / 15),
                'do_distance_confidence': 1.0 / (1.0 + do_distance / 20),
                'do_high_distance_confidence': 1.0 / (1.0 + do_high_distance / 20),
                'ti_do_positioning_accuracy': ti_do_positioning_accuracy,
                'ti_high_freq_indicator': ti_high_freq_indicator,
                'fa_mi_positioning_accuracy': fa_mi_positioning_accuracy,
                'fa_mi_harmonic_distinction': fa_mi_harmonic_distinction,
            })

            do_do_high_ratio = do_energy / (do_high_energy + 1e-8)
            do_harmonic2_ratio = do_harmonic2 / (do_high_harmonic2 + 1e-8)
            do_do_high_precision = do_distance / (do_high_distance + 1e-8)

            features.update({
                'do_do_high_energy_ratio': do_do_high_ratio,
                'do_do_high_harmonic2_ratio': do_harmonic2_ratio,
                'do_do_high_precision': do_do_high_precision
            })

            sol_freq = self.note_frequencies['sol']
            mi_freq = self.note_frequencies['mi']
            sol_energy = self._get_energy_around_freq(fft, freqs, sol_freq, 25)
            mi_energy = self._get_energy_around_freq(fft, freqs, mi_freq, 25)
            sol_mi_energy_ratio = sol_energy / (mi_energy + 1e-8)

            sol_harmonics = [self._get_energy_around_freq(fft, freqs, sol_freq * h, 30) for h in [2, 3, 4]]
            mi_harmonics = [self._get_energy_around_freq(fft, freqs, mi_freq * h, 30) for h in [2, 3, 4]]

            sol_harmonic_richness = sum(sol_harmonics) / (sol_energy + 1e-8)
            mi_harmonic_richness = sum(mi_harmonics) / (mi_energy + 1e-8)
            sol_mi_harmonic_ratio = sol_harmonic_richness / (mi_harmonic_richness + 1e-8)

            freq_gap = abs(sol_freq - mi_freq)
            measured_gap = abs(fundamental_freq - (sol_freq + mi_freq) / 2)
            gap_consistency = 1.0 / (1.0 + measured_gap / freq_gap)

            features.update({
                'sol_mi_energy_ratio': sol_mi_energy_ratio,
                'sol_mi_harmonic_ratio': sol_mi_harmonic_ratio,
                'sol_mi_gap_consistency': gap_consistency,
                'sol_distance': abs(fundamental_freq - sol_freq),
                'mi_distance': abs(fundamental_freq - mi_freq)
            })

            ti_freq = self.note_frequencies['ti']
            fa_freq = self.note_frequencies['fa']
            ti_energy = self._get_energy_around_freq(fft, freqs, ti_freq, 20)
            fa_energy = self._get_energy_around_freq(fft, freqs, fa_freq, 20)
            ti_fa_energy_ratio = ti_energy / (fa_energy + 1e-8)

            ti_harmonic3 = self._get_energy_around_freq(fft, freqs, ti_freq * 3, 40)
            fa_harmonic3 = self._get_energy_around_freq(fft, freqs, fa_freq * 3, 40)
            ti_fa_harmonic3_ratio = ti_harmonic3 / (fa_harmonic3 + 1e-8)

            high_freq_mask = freqs > 450
            ti_high_content = np.sum(fft[high_freq_mask])
            ti_high_ratio = ti_high_content / (total_energy + 1e-8)

            features.update({
                'ti_fa_energy_ratio': ti_fa_energy_ratio,
                'ti_fa_harmonic3_ratio': ti_fa_harmonic3_ratio,
                'ti_high_freq_ratio': ti_high_ratio,
                'ti_distance': abs(fundamental_freq - ti_freq),
                'fa_distance': abs(fundamental_freq - fa_freq)
            })

            spectral_centroid = np.sum(freqs[freqs > 0] * fft[freqs > 0]) / (np.sum(fft[freqs > 0]) + 1e-8)
            ti_centroid_match = 1.0 / (1.0 + abs(spectral_centroid - ti_freq * 1.5) / 100)
            do_high_centroid_match = 1.0 / (1.0 + abs(spectral_centroid - do_high_freq * 1.2) / 100)

            spectral_spread = np.sqrt(np.sum(((freqs[freqs > 0] - spectral_centroid) ** 2) * fft[freqs > 0]) / (
                    np.sum(fft[freqs > 0]) + 1e-8))
            ti_spread_match = spectral_spread / 200
            do_high_spread_match = 1.0 / (spectral_spread / 150 + 1.0)

            try:
                harmonic, percussive = librosa.effects.hpss(y)
                hnr = np.sum(harmonic ** 2) / (np.sum(percussive ** 2) + 1e-8)
                ti_hnr_match = 1.0 / (hnr + 1.0)
                do_high_hnr_match = hnr / (hnr + 1.0)
            except:
                ti_hnr_match = 0.5
                do_high_hnr_match = 0.5

            features.update({
                'ti_do_high_energy_ratio': ti_energy / (do_high_energy + 1e-8),
                'ti_centroid_match': ti_centroid_match,
                'do_high_centroid_match': do_high_centroid_match,
                'ti_spread_match': ti_spread_match,
                'do_high_spread_match': do_high_spread_match,
                'ti_hnr_match': ti_hnr_match,
                'do_high_hnr_match': do_high_hnr_match
            })

            re_fa_energy_ratio = re_energy / (fa_energy + 1e-8)
            re_fa_freq_gap = abs(re_freq - fa_freq)
            re_fa_measured_gap = abs(fundamental_freq - (re_freq + fa_freq) / 2)
            re_fa_gap_consistency = 1.0 / (1.0 + re_fa_measured_gap / re_fa_freq_gap)

            fa_harmonic2 = self._get_energy_around_freq(fft, freqs, fa_freq * 2, 35)
            re_harmonic_pattern = re_harmonic2 / (re_harmonic3 + 1e-8)
            fa_harmonic3 = self._get_energy_around_freq(fft, freqs, fa_freq * 3, 35)
            fa_harmonic_pattern = fa_harmonic2 / (fa_harmonic3 + 1e-8)
            re_fa_harmonic_pattern_ratio = re_harmonic_pattern / (fa_harmonic_pattern + 1e-8)

            try:
                rolloff_re_region = np.sum(fft[(freqs >= re_freq - 30) & (freqs <= re_freq + 100)])
                rolloff_fa_region = np.sum(fft[(freqs >= fa_freq - 30) & (freqs <= fa_freq + 100)])
                re_fa_rolloff_ratio = rolloff_re_region / (rolloff_fa_region + 1e-8)
            except:
                re_fa_rolloff_ratio = 1.0

            features.update({
                're_fa_energy_ratio': re_fa_energy_ratio,
                're_fa_gap_consistency': re_fa_gap_consistency,
                're_fa_harmonic_pattern_ratio': re_fa_harmonic_pattern_ratio,
                're_fa_rolloff_ratio': re_fa_rolloff_ratio
            })

            la_freq = self.note_frequencies['la']
            la_energy = self._get_energy_around_freq(fft, freqs, la_freq, 25)
            la_re_energy_ratio = la_energy / (re_energy + 1e-8)

            la_re_freq_gap = abs(la_freq - re_freq)
            la_re_measured_gap = abs(fundamental_freq - (la_freq + re_freq) / 2)
            la_re_gap_consistency = 1.0 / (1.0 + la_re_measured_gap / la_re_freq_gap)

            la_harmonic2 = self._get_energy_around_freq(fft, freqs, la_freq * 2, 40)
            la_harmonic3 = self._get_energy_around_freq(fft, freqs, la_freq * 3, 40)

            la_harmonic_richness = (la_harmonic2 + la_harmonic3) / (la_energy + 1e-8)
            re_harmonic_richness = (re_harmonic2 + re_harmonic3) / (re_energy + 1e-8)
            la_re_harmonic_richness_ratio = la_harmonic_richness / (re_harmonic_richness + 1e-8)

            max_amp = np.max(np.abs(y))
            if max_amp < 0.01:
                distance_compensation = 1.5
            else:
                distance_compensation = 1.0

            la_re_distance_compensated_ratio = la_re_energy_ratio * distance_compensation

            brightness_threshold = 400
            bright_energy = np.sum(fft[freqs > brightness_threshold])
            total_energy_for_brightness = np.sum(fft[freqs > 0])
            brightness_ratio = bright_energy / (total_energy_for_brightness + 1e-8)

            la_brightness_match = brightness_ratio if fundamental_freq > 350 else (1.0 - brightness_ratio)

            features.update({
                'la_re_energy_ratio': la_re_energy_ratio,
                'la_re_gap_consistency': la_re_gap_consistency,
                'la_re_harmonic_richness_ratio': la_re_harmonic_richness_ratio,
                'la_re_distance_compensated_ratio': la_re_distance_compensated_ratio,
                'la_brightness_match': la_brightness_match,
                'la_distance': abs(fundamental_freq - la_freq)
            })

        except Exception as e:
            logger.error(f"Note-specific feature analysis error: {e}")
            default_features = {}
            for key in [
                're_energy', 're_do_energy_ratio', 're_do_high_energy_ratio', 're_distance', 'do_distance',
                'do_high_distance', 're_positioning_score', 'do_positioning_score', 'do_high_positioning_score',
                're_freq_isolation', 're_harmonic_ratio', 're_harmonic2_energy', 're_harmonic3_energy',
                're_peak_sharpness', 're_centroid_match', 're_fundamental_strength', 're_distance_confidence',
                'do_distance_confidence', 'do_high_distance_confidence', 'do_do_high_energy_ratio',
                'do_do_high_harmonic2_ratio', 'do_do_high_precision', 'sol_mi_energy_ratio',
                'sol_mi_harmonic_ratio', 'sol_mi_gap_consistency', 'sol_distance', 'mi_distance',
                'ti_fa_energy_ratio', 'ti_fa_harmonic3_ratio', 'ti_high_freq_ratio', 'ti_distance', 'fa_distance',
                'ti_do_high_energy_ratio', 'ti_centroid_match', 'do_high_centroid_match', 'ti_spread_match',
                'do_high_spread_match', 'ti_hnr_match', 'do_high_hnr_match', 're_fa_energy_ratio',
                're_fa_gap_consistency', 're_fa_harmonic_pattern_ratio', 're_fa_rolloff_ratio', 'la_re_energy_ratio',
                'la_re_gap_consistency', 'la_re_harmonic_richness_ratio', 'la_re_distance_compensated_ratio',
                'la_brightness_match', 'la_distance'
            ]:
                default_features[key] = 0.5
            features.update(default_features)

        return features

    def _get_energy_around_freq(self, fft, freqs, target_freq, tolerance):
        mask = (freqs >= target_freq - tolerance) & (freqs <= target_freq + tolerance)
        if np.any(mask):
            return float(np.sum(fft[mask]))
        return 0.0

    def extract_mfcc_features(self, y, sr=None):
        if sr is None:
            sr = self.sr

        if len(y) == 0 or np.max(np.abs(y)) == 0:
            return np.zeros((self.n_mfcc * 2, self.target_frames))

        # Detect duration type for appropriate processing
        duration_type = self.detect_note_duration_type(y, sr)
        y = self.advanced_distance_normalization(y, duration_type)

        try:
            nyquist = sr / 2
            max_amp = np.max(np.abs(y))

            # Adjust filtering based on duration and amplitude
            if duration_type == 'short':
                if max_amp < 0.01:
                    low_freq = 100 / nyquist
                    high_freq = min(1200 / nyquist, 0.99)
                else:
                    low_freq = 120 / nyquist
                    high_freq = min(1000 / nyquist, 0.99)
            else:
                # Original long note filtering
                if max_amp < 0.01:
                    low_freq = 120 / nyquist
                    high_freq = min(1000 / nyquist, 0.99)
                elif max_amp > 0.5:
                    low_freq = 180 / nyquist
                    high_freq = min(700 / nyquist, 0.99)
                else:
                    low_freq = 150 / nyquist
                    high_freq = min(850 / nyquist, 0.99)

            b, a = butter(4, [low_freq, high_freq], btype='band')
            y = lfilter(b, a, y)

        except Exception as e:
            pass

        try:
            max_amp = np.max(np.abs(y))

            # Duration-specific MFCC parameters
            if duration_type == 'short':
                if max_amp < 0.01:
                    power = 1.3
                    fmin = 80
                    fmax = 1400
                else:
                    power = 1.8
                    fmin = 100
                    fmax = 1200
            else:
                # Original long note parameters
                if max_amp < 0.01:
                    power = 1.5
                    fmin = 100
                    fmax = 1200
                elif max_amp > 0.5:
                    power = 2.5
                    fmin = 150
                    fmax = 800
                else:
                    power = 2.0
                    fmin = 120
                    fmax = 1000

            mfccs = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length,
                fmin=fmin, fmax=fmax, window='hann', power=power
            )

            delta_mfccs = librosa.feature.delta(mfccs)
            combined_features = np.vstack([mfccs, delta_mfccs])
            combined_features = self._pad_or_truncate(combined_features, self.target_frames)

            return combined_features

        except Exception as e:
            return np.zeros((self.n_mfcc * 2, self.target_frames))

    def _pad_or_truncate(self, features, target_frames):
        current_frames = features.shape[1]

        if current_frames > target_frames:
            start_idx = (current_frames - target_frames) // 2
            return features[:, start_idx:start_idx + target_frames]
        elif current_frames < target_frames:
            pad_width = target_frames - current_frames
            pad_left = int(pad_width // 2)
            pad_right = int(pad_width - pad_left)
            return np.pad(features, ((0, 0), (pad_left, pad_right)), mode='constant')
        else:
            return features

    def extract_enhanced_features(self, y, sr=None):
        if sr is None:
            sr = self.sr

        max_amplitude = np.max(np.abs(y))
        rms_amplitude = np.sqrt(np.mean(y ** 2))

        duration = len(y) / sr
        # FIXED: Much more lenient silence thresholds for 1-second recordings
        if duration < 1.5:  # Short recording
            silence_threshold = 0.0008  # More lenient (was 0.002)
            rms_threshold = 0.0003  # More lenient (was 0.001)
        else:
            silence_threshold = 0.001
            rms_threshold = 0.0005

        # FIXED: Only return no_angklung features for truly silent recordings
        if (len(y) == 0 or max_amplitude < silence_threshold or
                (rms_amplitude < rms_threshold and max_amplitude < 0.0005)):
            features = np.zeros(96)
            features[0] = 0.0
            features[22] = 0.0
            features[23] = 0.0
            return features

        # Detect duration type
        duration_type = self.detect_note_duration_type(y, sr)
        y = self.advanced_distance_normalization(y, duration_type)
        features = []

        try:
            # 1. Fundamental frequency analysis (11 features)
            fundamental_freq = self.extract_fundamental_with_duration_awareness(y, sr, duration_type)

            note_distances = {}
            for note, freq in self.note_frequencies.items():
                note_distances[note] = abs(fundamental_freq - freq)

            sorted_distances = sorted(note_distances.items(), key=lambda x: x[1])
            closest_note = sorted_distances[0][0]
            closest_distance = sorted_distances[0][1]
            second_closest_distance = sorted_distances[1][1] if len(sorted_distances) > 1 else 100

            features.extend([
                fundamental_freq,
                closest_distance,
                closest_distance / (second_closest_distance + 1e-8),
                note_distances['do'],
                note_distances['re'],
                note_distances['mi'],
                note_distances['fa'],
                note_distances['sol'],
                note_distances['la'],
                note_distances['ti'],
                note_distances['do_high']
            ])

            # 2. Duration-specific features (11 features) - NEW!
            duration_features = self.extract_duration_specific_features(y, sr, fundamental_freq, duration_type)
            duration_feature_values = [
                duration_features.get('effective_duration', 1.0),
                duration_features.get('onset_time', 0.05),
                duration_features.get('sustain_ratio', 0.8),
                duration_features.get('attack_ratio', 0.3),
                duration_features.get('peak_attack_time', 0.02),
                duration_features.get('decay_ratio', 0.2),
                duration_features.get('decay_slope', -0.1),
                duration_features.get('harmonic_development', 1.0),
                duration_features.get('spectral_flux_mean', 0.1),
                duration_features.get('spectral_flux_std', 0.05),
                duration_features.get('duration_type_numeric', 0.5)
            ]
            features.extend(duration_feature_values)

            # 3. Non-angklung discrimination features (5 features)
            fft = np.abs(np.fft.fft(y))
            freqs = np.fft.fftfreq(len(y), 1 / sr)

            in_angklung_range = 1.0 if (220 <= fundamental_freq <= 580) else 0.0

            total_energy = np.sum(fft[freqs > 0])
            angklung_band_energy = np.sum(fft[(freqs >= 220) & (freqs <= 580)])
            angklung_energy_ratio = angklung_band_energy / (total_energy + 1e-8)

            positive_freqs = freqs[freqs > 0]
            positive_fft = fft[freqs > 0]
            if len(positive_freqs) > 0:
                spectral_centroid = np.sum(positive_freqs * positive_fft) / (np.sum(positive_fft) + 1e-8)
                spectral_spread = np.sqrt(np.sum(((positive_freqs - spectral_centroid) ** 2) * positive_fft) /
                                          (np.sum(positive_fft) + 1e-8))
            else:
                spectral_centroid = 0
                spectral_spread = 0

            harmonic_regularity = 0.0
            if fundamental_freq > 0:
                expected_harmonics = [fundamental_freq * h for h in [1, 2, 3, 4, 5]]
                harmonic_energies = []
                for h_freq in expected_harmonics:
                    if h_freq < sr / 2:
                        h_energy = self._get_energy_around_freq(fft, freqs, h_freq, 20)
                        harmonic_energies.append(h_energy)

                if len(harmonic_energies) >= 3:
                    harmonic_energies = np.array(harmonic_energies)
                    if harmonic_energies[0] > 0:
                        harmonic_ratios = harmonic_energies[1:] / harmonic_energies[0]
                        expected_pattern = np.array([0.5, 0.3, 0.2, 0.1])[:len(harmonic_ratios)]
                        harmonic_regularity = 1.0 / (1.0 + np.mean(np.abs(harmonic_ratios - expected_pattern)))

            features.extend([
                in_angklung_range,
                angklung_energy_ratio,
                spectral_spread / 1000,
                harmonic_regularity,
                1.0 if spectral_centroid > 0 else 0.0
            ])

            # 4. Note-specific features (19 features) - with duration awareness
            note_features = self.analyze_note_specific_features(y, sr, fundamental_freq, duration_type)

            re_specific_features = [
                note_features.get('re_energy', 1.0),
                note_features.get('re_do_energy_ratio', 1.0),
                note_features.get('re_do_high_energy_ratio', 1.0),
                note_features.get('re_positioning_score', 0.5),
                note_features.get('do_positioning_score', 0.5),
                note_features.get('do_high_positioning_score', 0.5),
                note_features.get('re_freq_isolation', 0.5),
                note_features.get('re_harmonic_ratio', 1.0),
                note_features.get('re_harmonic2_energy', 1.0),
                note_features.get('re_harmonic3_energy', 1.0),
                note_features.get('re_peak_sharpness', 1.0),
                note_features.get('re_centroid_match', 0.5),
                note_features.get('re_fundamental_strength', 0.3),
                note_features.get('re_distance_confidence', 0.5),
                note_features.get('do_distance_confidence', 0.5),
                note_features.get('do_high_distance_confidence', 0.5),
                note_features.get('re_distance', 50.0),
                note_features.get('do_distance', 50.0),
                note_features.get('do_high_distance', 50.0)
            ]
            features.extend(re_specific_features)
            # 4.5. Short duration discriminative features (7 features) - NEW!
            if duration_type == 'short':
                short_features = self._extract_short_duration_discriminative_features(y, sr, fundamental_freq)
                short_feature_values = [
                    short_features.get('sol_la_energy_ratio', 1.0),
                    short_features.get('fa_mi_energy_ratio', 1.0),
                    short_features.get('sol_precision_score', 0.5),
                    short_features.get('la_precision_score', 0.5),
                    short_features.get('fa_precision_score', 0.5),
                    short_features.get('mi_precision_score', 0.5),
                    short_features.get('frequency_specificity', 0.0)
                ]
                features.extend(short_feature_values)
            else:
                # For long notes, add default values
                features.extend([1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.0])

            # Update the total feature count
            features = features[:103]  # Changed from 96 to 103 (96 + 7 new features)
            while len(features) < 103:
                features.append(0.0)
            # 5. Continue with remaining spectral and harmonic features (50 features)
            # Spectral features (8 features)
            try:
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                centroid_mean = np.mean(centroid)
                centroid_std = np.std(centroid)

                rolloff_85 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
                rolloff_95 = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)[0]
                rolloff_mean = np.mean(rolloff_85)
                rolloff_ratio = np.mean(rolloff_85) / (np.mean(rolloff_95) + 1e-8)

                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
                bandwidth_mean = np.mean(bandwidth)

                zcr = librosa.feature.zero_crossing_rate(y)[0]
                zcr_mean = np.mean(zcr)

                features.extend([
                    centroid_mean / (fundamental_freq + 1e-8),
                    centroid_std / (centroid_mean + 1e-8),
                    rolloff_mean / (fundamental_freq + 1e-8),
                    rolloff_ratio,
                    bandwidth_mean / (fundamental_freq + 1e-8),
                    zcr_mean,
                    centroid_mean / 1000,
                    bandwidth_mean / 1000
                ])
            except:
                features.extend([2.0, 0.1, 1.5, 0.8, 1.0, 0.1, 0.4, 0.3])

            # Harmonic analysis (15 features) - duration-aware
            try:
                max_amp = np.max(np.abs(y))
                tolerance = 25 if duration_type == 'short' else 30

                harmonic_energies = []
                for h in [1, 2, 3, 4, 5]:
                    harmonic_freq = fundamental_freq * h
                    energy = self._get_energy_around_freq(fft, freqs, harmonic_freq, tolerance)
                    harmonic_energies.append(energy)

                fundamental_energy = harmonic_energies[0] if harmonic_energies[0] > 0 else 1.0
                harmonic_ratios = [e / fundamental_energy for e in harmonic_energies[1:]]

                total_harmonic = sum(harmonic_energies[1:])
                harmonic_richness = total_harmonic / fundamental_energy

                odd_harmonics = harmonic_energies[0] + harmonic_energies[2] + harmonic_energies[4]
                even_harmonics = harmonic_energies[1] + harmonic_energies[3]
                odd_even_ratio = odd_harmonics / (even_harmonics + 1e-8)

                if len(harmonic_ratios) >= 2:
                    decay_rate = harmonic_ratios[0] / (harmonic_ratios[1] + 1e-8)
                else:
                    decay_rate = 1.0

                re_freq = self.note_frequencies['re']
                re_2nd_harmonic = self._get_energy_around_freq(fft, freqs, re_freq * 2, tolerance)
                re_3rd_harmonic = self._get_energy_around_freq(fft, freqs, re_freq * 3, tolerance)
                re_harmonic_signature = re_2nd_harmonic / (re_3rd_harmonic + 1e-8)

                features.extend([
                    harmonic_ratios[0] if len(harmonic_ratios) > 0 else 0.5,
                    harmonic_ratios[1] if len(harmonic_ratios) > 1 else 0.3,
                    harmonic_ratios[2] if len(harmonic_ratios) > 2 else 0.2,
                    harmonic_ratios[3] if len(harmonic_ratios) > 3 else 0.1,
                    harmonic_richness,
                    odd_even_ratio,
                    decay_rate,
                    fundamental_energy / (np.sum(fft) + 1e-8),
                    np.std(harmonic_ratios) if len(harmonic_ratios) > 1 else 0.1,
                    len([r for r in harmonic_ratios if r > 0.1]),
                    re_harmonic_signature,
                    re_2nd_harmonic / (fundamental_energy + 1e-8),
                    re_3rd_harmonic / (fundamental_energy + 1e-8),
                    (re_2nd_harmonic + re_3rd_harmonic) / (fundamental_energy + 1e-8),
                    np.max(harmonic_energies) / (np.mean(harmonic_energies) + 1e-8)
                ])
            except:
                features.extend([0.5, 0.3, 0.2, 0.1, 1.0, 1.5, 2.0, 0.3, 0.1, 2.0, 1.0, 0.3, 0.2, 0.5, 2.0])

            # Remaining note confusion features (30 features) - abbreviated for space
            # Include all the original confusion analysis features here
            default_confusion_features = [1.0] * 30  # Placeholder - implement full feature set
            features.extend(default_confusion_features)

        except Exception as e:
            features = [350.0] * 96

        features = features[:96]  # Changed from 85 to 96 for duration features
        while len(features) < 96:
            features.append(0.0)

        feature_array = np.array(features, dtype=np.float32)
        return np.nan_to_num(feature_array, nan=0.0, posinf=1000.0, neginf=-1000.0)

    def prepare_enhanced_input(self, y, sr=None):
        mfcc_features = self.extract_mfcc_features(y, sr)
        enhanced_features = self.extract_enhanced_features(y, sr)
        return mfcc_features, enhanced_features

    def fit_scaler(self, enhanced_features_list):
        if len(enhanced_features_list) > 0:
            features_array = np.vstack(enhanced_features_list)
            self.scaler.fit(features_array)
            self.scaler_fitted = True

    def transform_enhanced_features(self, features):
        if not self.scaler_fitted:
            raise ValueError("Scaler not fitted!")
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
            return self.scaler.transform(features)[0]
        return self.scaler.transform(features)


def load_multi_duration_model():
    """Load model with minimal changes for compatibility"""
    global model, extractor, label_encoder, metadata

    try:
        model_path = 'model/unified_angklung_model.keras'
        extractor_path = 'model/unified_angklung_extractor.pkl'
        label_encoder_path = 'model/unified_angklung_label_encoder.pkl'
        metadata_path = 'model/unified_angklung_metadata.pkl'

        if not all(os.path.exists(path) for path in [model_path, extractor_path, label_encoder_path]):
            missing_files = [path for path in [model_path, extractor_path, label_encoder_path]
                             if not os.path.exists(path)]
            logger.error(f"Missing unified model files: {missing_files}")
            return False

        # Load model normally
        model = keras.models.load_model(model_path)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        logger.info("Unified CNN model loaded")

        with open(extractor_path, 'rb') as f:
            extractor = pickle.load(f)

        # Don't modify extractor methods - use as-is
        logger.info("Multi-duration extractor loaded (unmodified)")

        with open(label_encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        logger.info("Label encoder loaded")

        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            logger.info("Multi-duration metadata loaded")
        else:
            metadata = {
                'labels': ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti', 'do_high', 'no_angklung'],
                'model_type': 'Multi-Duration-Angklung-CNN'
            }

        # Warm up with original methods
        dummy_audio = np.random.randn(22050) * 0.1
        test_mfcc, test_enhanced = extractor.prepare_enhanced_input(dummy_audio)

        mfcc_input = test_mfcc[np.newaxis, ...]
        enhanced_scaled = extractor.transform_enhanced_features(test_enhanced)
        enhanced_input = enhanced_scaled[np.newaxis, ...]

        test_pred = model.predict([mfcc_input, enhanced_input], verbose=0)

        logger.info("✓ Original pipeline working correctly")
        logger.info(f"Model type: {metadata.get('model_type', 'Unknown')}")

        return True

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False


def predict_angklung_multi_duration(audio_path):
    """Enhanced prediction function for multi-duration model"""
    global model, extractor, label_encoder, metadata

    try:
        signal, sr = librosa.load(audio_path, sr=22050, mono=True)

        duration = len(signal) / sr
        original_max_amp = float(np.max(np.abs(signal)))
        original_rms_amp = float(np.sqrt(np.mean(signal ** 2)))

        if duration < 1.2:  # Short recording
            if original_max_amp < 0.005 or original_rms_amp < 0.002:
                return {
                    'success': True,
                    'prediction': 'no_angklung',
                    'confidence': 0.90,
                    'note_detected': False,
                    'reason': 'Signal too weak for short recording - likely silence'
                }

        # Extract features FIRST before using fundamental_freq
        mfcc_features, enhanced_features = extractor.prepare_enhanced_input(signal, sr)

        # NOW we can get fundamental_freq from enhanced_features
        fundamental_freq = float(enhanced_features[0])

        # Now this check can work properly
        if fundamental_freq < 200 or fundamental_freq > 600:
            if duration < 1.5:  # For short recordings, be more strict
                return {
                    'success': True,
                    'prediction': 'no_angklung',
                    'confidence': 0.85,
                    'reason': 'Frequency outside angklung range in short recording'
                }

        # Distance estimation
        if original_max_amp < 0.005:
            distance_estimate = "very_far"
            distance_note = "Phone very far from speaker"
        elif original_max_amp < 0.05:
            distance_estimate = "far"
            distance_note = "Phone far from speaker"
        elif original_max_amp > 0.7:
            distance_estimate = "very_near"
            distance_note = "Phone very close to speaker"
        elif original_max_amp > 0.3:
            distance_estimate = "near"
            distance_note = "Phone close to speaker"
        else:
            distance_estimate = "normal"
            distance_note = "Phone at normal distance from speaker"

        # Extract features
        mfcc_features, enhanced_features = extractor.prepare_enhanced_input(signal, sr)

        # Get enhanced features (should be 96 for multi-duration model)
        expected_features = 96
        if len(enhanced_features) != expected_features:
            logger.warning(f"Feature count mismatch: got {len(enhanced_features)}, expected {expected_features}")

        fundamental_freq = float(enhanced_features[0])

        # Duration detection
        duration_type = extractor.detect_note_duration_type(signal, sr)
        duration_features = {
            'effective_duration': float(enhanced_features[11]),
            'onset_time': float(enhanced_features[12]),
            'sustain_ratio': float(enhanced_features[13]),
            'attack_ratio': float(enhanced_features[14]),
            'peak_attack_time': float(enhanced_features[15]),
            'decay_ratio': float(enhanced_features[16]),
            'decay_slope': float(enhanced_features[17]),
            'harmonic_development': float(enhanced_features[18]),
            'spectral_flux_mean': float(enhanced_features[19]),
            'spectral_flux_std': float(enhanced_features[20]),
            'duration_type_numeric': float(enhanced_features[21])
        }

        # Negative detection features
        in_angklung_range = float(enhanced_features[22])
        angklung_energy_ratio = float(enhanced_features[23])
        spectral_spread_norm = float(enhanced_features[24])
        harmonic_regularity = float(enhanced_features[25])
        has_spectral_content = float(enhanced_features[26])

        # Scale enhanced features
        enhanced_features_scaled = extractor.transform_enhanced_features(enhanced_features)

        # Prepare model inputs
        mfcc_input = mfcc_features[np.newaxis, ...]
        enhanced_input = enhanced_features_scaled[np.newaxis, ...]

        # Make prediction
        predictions = model.predict([mfcc_input, enhanced_input], verbose=0)[0]

        predicted_class = np.argmax(predictions)
        confidence = float(predictions[predicted_class])

        # Get labels
        if hasattr(label_encoder, 'classes_'):
            labels = label_encoder.classes_
        else:
            labels = metadata.get('labels', ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti', 'do_high', 'no_angklung'])

        predicted_note = labels[predicted_class]

        # Build predictions dictionary
        all_predictions = {}
        for i, label in enumerate(labels):
            all_predictions[label] = float(predictions[i]) if i < len(predictions) else 0.0

        sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)

        # Confidence assessment
        if confidence >= 0.9:
            confidence_level = "very_high"
        elif confidence >= 0.75:
            confidence_level = "high"
        elif confidence >= 0.6:
            confidence_level = "medium"
        elif confidence >= 0.4:
            confidence_level = "low"
        else:
            confidence_level = "very_low"

        prediction_std = float(np.std(predictions))
        top_2_diff = float(sorted_predictions[0][1] - sorted_predictions[1][1]) if len(sorted_predictions) >= 2 else 1.0

        # Reliability assessment
        reliability_issues = []

        if duration < 0.3:
            reliability_issues.append("Audio too short")
        elif duration > 15:
            reliability_issues.append("Audio very long")

        if original_max_amp < 0.0001:
            reliability_issues.append("Audio too quiet")
        elif original_max_amp > 0.99:
            reliability_issues.append("Audio may be clipping")

        if confidence < 0.35:
            reliability_issues.append("Low confidence")

        if top_2_diff < 0.15:
            reliability_issues.append("Close competing predictions")

        freq_valid = 220 <= fundamental_freq <= 580
        if not freq_valid and predicted_note != 'no_angklung':
            reliability_issues.append("Fundamental frequency outside Angklung range")

        # Negative detection analysis
        negative_detection_analysis = {}
        if 'no_angklung' in labels:
            no_angklung_confidence = all_predictions.get('no_angklung', 0.0)
            is_negative_prediction = predicted_note == 'no_angklung'

            negative_detection_analysis = {
                'negative_class_available': True,
                'no_angklung_confidence': float(no_angklung_confidence),
                'is_negative_prediction': bool(is_negative_prediction),
                'negative_indicators': {
                    'frequency_outside_range': not freq_valid,
                    'low_angklung_energy': angklung_energy_ratio < 0.3,
                    'high_spectral_spread': spectral_spread_norm > 0.7,
                    'low_harmonic_regularity': harmonic_regularity < 0.3,
                    'no_spectral_content': has_spectral_content < 0.5
                },
                'discrimination_features': {
                    'in_angklung_range': float(in_angklung_range),
                    'angklung_energy_ratio': float(angklung_energy_ratio),
                    'spectral_spread_normalized': float(spectral_spread_norm),
                    'harmonic_regularity': float(harmonic_regularity),
                    'has_spectral_content': float(has_spectral_content)
                }
            }

            # Add specific negative detection reasoning
            if is_negative_prediction:
                reasons = []
                if not freq_valid:
                    reasons.append("Fundamental frequency outside angklung range")
                if angklung_energy_ratio < 0.3:
                    reasons.append("Low energy in angklung frequency band")
                if spectral_spread_norm > 0.7:
                    reasons.append("Energy too spread across frequencies")
                if harmonic_regularity < 0.3:
                    reasons.append("Lacks regular harmonic structure")
                if has_spectral_content < 0.5:
                    reasons.append("Insufficient spectral content")
                if original_max_amp < 0.001:
                    reasons.append("Signal too quiet (likely silence)")

                negative_detection_analysis['negative_reasons'] = reasons
        else:
            negative_detection_analysis = {
                'negative_class_available': False,
                'note': '8-class model - no negative detection'
            }

        is_reliable = (len(reliability_issues) <= 1 and
                       confidence >= 0.35 and
                       top_2_diff >= 0.10)

        # Note type analysis
        note_type = duration_type
        note_type_confidence = 0.8  # High confidence for duration detection

        # Model info
        model_type = metadata.get('model_type', 'Unknown')
        is_multi_duration = 'Multi-Duration' in model_type

        result = {
            'success': True,
            'prediction': str(predicted_note),
            'confidence': float(confidence),
            'confidence_level': str(confidence_level),
            'all_predictions': {k: float(v) for k, v in all_predictions.items()},
            'top_3_predictions': [
                {'note': str(note), 'confidence': float(prob)}
                for note, prob in sorted_predictions[:3]
            ],
            'reliability_assessment': {
                'is_reliable': bool(is_reliable),
                'issues': [str(issue) for issue in reliability_issues],
                'overall_quality': str(confidence_level)
            },
            'audio_analysis': {
                'duration_seconds': float(duration),
                'original_max_amplitude': float(original_max_amp),
                'original_rms_amplitude': float(original_rms_amp),
                'sample_rate': int(sr),
                'distance_estimate': str(distance_estimate),
                'distance_level': distance_note,
                'detected_duration_type': str(duration_type)
            },
            'negative_detection': convert_to_serializable(negative_detection_analysis),
            'model_info': {
                'model_type': str(model_type),
                'version': 'Multi-Duration-v1.0',
                'classes_supported': len(labels),
                'negative_detection_enabled': bool('no_angklung' in labels),
                'mfcc_features_shape': [int(x) for x in mfcc_features.shape],
                'enhanced_features_count': int(len(enhanced_features)),
                'prediction_distribution_std': float(prediction_std),
                'top_2_difference': float(top_2_diff)
            },
            'feature_diagnostics': {
                'fundamental_frequency': float(fundamental_freq),
                'frequency_range_valid': bool(freq_valid),
                'duration_features': duration_features,
                'duration_type': str(duration_type)
            },
            'note_type': note_type,
            'note_type_confidence': float(note_type_confidence),
            'duration_analysis': {
                'audio_duration': float(duration),
                'predicted_type': note_type,
                'type_confidence': float(note_type_confidence),
                'detection_method': 'duration_aware_extraction'
            }
        }

        return convert_to_serializable(result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }

 # 3. Optimize only the prediction function - keep feature extraction intact:
def predict_angklung_optimized_conservative(audio_path):
        """Conservative optimization - only change non-critical parts"""
        global model, extractor, label_encoder, metadata

        try:
            # OPTIMIZATION 1: Faster audio loading
            signal, sr = librosa.load(audio_path, sr=22050, mono=True, duration=2.5)  # Limit duration slightly

            duration = len(signal) / sr
            original_max_amp = float(np.max(np.abs(signal)))
            original_rms_amp = float(np.sqrt(np.mean(signal ** 2)))

            print(f"Audio loaded: duration={duration:.2f}s, max_amp={original_max_amp:.6f}")

            # OPTIMIZATION 2: Quick early exits only for truly silent audio
            if duration < 0.3:  # Very short recordings
                return {
                    'success': True,
                    'prediction': 'no_angklung',
                    'confidence': 0.90,
                    'reason': 'Audio too short'
                }

            if original_max_amp < 0.00001:  # Truly silent
                return {
                    'success': True,
                    'prediction': 'no_angklung',
                    'confidence': 0.95,
                    'reason': 'No audio detected'
                }

            # KEEP ORIGINAL FEATURE EXTRACTION - don't modify
            mfcc_features, enhanced_features = extractor.prepare_enhanced_input(signal, sr)

            fundamental_freq = float(enhanced_features[0])
            print(f"Fundamental frequency detected: {fundamental_freq:.2f} Hz")

            # OPTIMIZATION 3: Skip detailed analysis, go straight to prediction
            duration_type = extractor.detect_note_duration_type(signal, sr)

            # KEEP ORIGINAL SCALING
            enhanced_features_scaled = extractor.transform_enhanced_features(enhanced_features)

            # KEEP ORIGINAL MODEL INFERENCE
            mfcc_input = mfcc_features[np.newaxis, ...]
            enhanced_input = enhanced_features_scaled[np.newaxis, ...]

            predictions = model.predict([mfcc_input, enhanced_input], verbose=0, batch_size=1)[0]

            predicted_class = np.argmax(predictions)
            confidence = float(predictions[predicted_class])

            if hasattr(label_encoder, 'classes_'):
                labels = label_encoder.classes_
            else:
                labels = metadata.get('labels', ['do', 're', 'mi', 'fa', 'sol', 'la', 'ti', 'do_high', 'no_angklung'])

            predicted_note = labels[predicted_class]

            # Debug output
            print(f"Predictions: {dict(zip(labels[:len(predictions)], predictions))}")
            print(f"Final prediction: {predicted_note} ({confidence:.4f})")

            all_predictions = {}
            for i, label in enumerate(labels):
                all_predictions[label] = float(predictions[i]) if i < len(predictions) else 0.0

            sorted_predictions = sorted(all_predictions.items(), key=lambda x: x[1], reverse=True)
            confidence_level = "high" if confidence >= 0.7 else "medium" if confidence >= 0.5 else "low"

            # OPTIMIZATION 4: Minimal response structure
            result = {
                'success': True,
                'prediction': str(predicted_note),
                'confidence': float(confidence),
                'confidence_level': str(confidence_level),
                'all_predictions': {k: float(v) for k, v in all_predictions.items()},
                'top_3_predictions': [
                    {'note': str(note), 'confidence': float(prob)}
                    for note, prob in sorted_predictions[:3]
                ],
                'audio_analysis': {
                    'duration_seconds': float(duration),
                    'detected_duration_type': str(duration_type),
                    'fundamental_frequency': float(fundamental_freq)
                },
                'model_info': {
                    'model_type': 'Conservative-Optimized',
                    'version': 'v1.0'
                }
            }

            return convert_to_serializable(result)

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': f'Prediction error: {str(e)}'
            }

@app.route('/simple_test', methods=['GET'])
def simple_test():
    """Ultra-simple test to verify Flutter can read boolean values"""
    response_data = {
        'success': True,
        'supports_short_notes': True,
        'test_boolean': True,
        'test_string': 'true',
        'test_number': 1
    }

    logger.info(f"Simple test sending: {response_data}")
    return jsonify(response_data)


@app.route('/test_connection', methods=['GET'])
def test_connection():
    try:
        model_loaded = model is not None
        extractor_loaded = extractor is not None

        labels = metadata.get('labels', []) if metadata else []
        model_type = metadata.get('model_type', 'Unknown') if metadata else 'Unknown'

        # UPDATED: Detect unified model instead of multi-duration
        is_unified_model = 'Unified' in model_type or 'Multi-Duration' in model_type

        # Unified model always supports short notes
        supports_short_notes = is_unified_model

        negative_detection_enabled = 'no_angklung' in labels

        logger.info(f"Model type detected: {model_type}")
        logger.info(f"Is unified model: {is_unified_model}")
        logger.info(f"Labels: {labels}")
        logger.info(f"Short notes support: {supports_short_notes}")

        result = {
            'success': True,
            'message': f'Angklung {"Unified" if is_unified_model else "Standard"} CNN Server is running!',
            'version': 'unified-v1.0' if is_unified_model else 'standard-v1.0',
            'model_type': model_type,
            'model_loaded': bool(model_loaded),
            'extractor_loaded': bool(extractor_loaded),
            'supported_notes': labels,
            'classes_supported': len(labels),
            'negative_detection_enabled': bool(negative_detection_enabled),
            'supports_short_notes': bool(supports_short_notes),
            'is_cnn_mfcc_model': True,

            'features': {
                'enhanced_features_count': 96 if is_unified_model else 85,
                'duration_aware_features': is_unified_model,
                'distance_compensation': True,
                'negative_sample_detection': negative_detection_enabled,
                'cnn_architecture': True,
                'mfcc_features': True,
                'short_notes_support': supports_short_notes
            },

            'improvements_active': {
                'cnn_mfcc_model': True,
                'enhanced_features': True,
                'distance_normalization': True,
                'short_notes': supports_short_notes,
                'negative_detection': negative_detection_enabled,
                'duration_aware_processing': is_unified_model
            }
        }

        # Add unified model features
        if is_unified_model:
            result['unified_features'] = {
                'duration_detection': 'ACTIVE - Automatically detects short vs long notes',
                'duration_features': '11 new duration-specific features',
                'total_features': '96 (85 original + 11 duration features)',
                'supported_classes': 'do, re, mi, fa, sol, la, ti, do_high, no_angklung',
                'unified_processing': 'ACTIVE - No duration separation needed'
            }
            result['short_notes_accuracy'] = 0.85
            result['expected_accuracy'] = 0.88

        return jsonify(convert_to_serializable(result))

    except Exception as e:
        logger.error(f"test_connection error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/debug_connection', methods=['GET'])
def debug_connection():
    """Debug endpoint to see exactly what Flutter receives"""
    try:
        model_loaded = model is not None
        extractor_loaded = extractor is not None

        labels = metadata.get('labels', []) if metadata else []
        model_type = metadata.get('model_type', 'Unknown') if metadata else 'Unknown'

        is_multi_duration = 'Multi-Duration' in model_type

        # EXPLICIT boolean conversion
        supports_short_notes = True if is_multi_duration else False
        negative_detection_enabled = 'no_angklung' in labels

        # Create the simplest possible response
        result = {
            'success': True,
            'supports_short_notes': supports_short_notes,
            'is_multi_duration': is_multi_duration,
            'model_type': model_type,
            'labels': labels,
            'label_count': len(labels),
            'negative_detection_enabled': negative_detection_enabled,
            'improvements_active': {
                'short_notes': supports_short_notes
            }
        }

        # Log what we're about to send
        logger.info("=== DEBUG RESPONSE ===")
        logger.info(f"Raw result: {result}")
        logger.info(f"supports_short_notes type: {type(result['supports_short_notes'])}")
        logger.info(f"supports_short_notes value: {result['supports_short_notes']}")

        # Convert and log again
        converted_result = convert_to_serializable(result)
        logger.info(f"Converted result: {converted_result}")
        logger.info(f"Converted supports_short_notes type: {type(converted_result['supports_short_notes'])}")
        logger.info(f"Converted supports_short_notes value: {converted_result['supports_short_notes']}")

        return jsonify(converted_result)

    except Exception as e:
        logger.error(f"Debug endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict_note', methods=['POST'])
def predict_note():
    try:
        if model is None or extractor is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded properly'
            }), 500

        if 'audio' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No audio file provided'
            }), 400

        audio_file = request.files['audio']
        if audio_file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No audio file selected'
            }), 400

        temp_path = f"temp_angklung_{os.getpid()}.wav"

        try:
            audio_file.save(temp_path)

            file_size = os.path.getsize(temp_path)
            if file_size < 500:
                return jsonify({
                    'success': False,
                    'error': 'Audio file too small'
                })

            # Use conservative optimization - this function should now exist
            result = predict_angklung_optimized_conservative(temp_path)
            return jsonify(convert_to_serializable(result))

        finally:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass

    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/model_status', methods=['GET'])
def model_status():
    """Detailed model status and capabilities"""
    try:
        # Current model info
        current_model_info = None
        if model and metadata:
            current_model_info = {
                'type': metadata.get('model_type', 'Unknown'),
                'version': metadata.get('version', 'Unknown'),
                'labels': metadata.get('labels', []),
                'short_notes_capability': metadata.get('short_notes_capability', False),
                'classes_count': len(metadata.get('labels', [])),
                'negative_detection': 'no_angklung' in metadata.get('labels', [])
            }

        result = {
            'success': True,
            'current_model_loaded': model is not None,
            'current_model_info': current_model_info,
            'multi_duration_model': 'Multi-Duration' in metadata.get('model_type', '') if metadata else False,
            'recommendations': []
        }

        # Add recommendations
        if current_model_info and current_model_info.get('type', '').startswith('Multi-Duration'):
            result['recommendations'].append("Multi-duration model active! Supports both short and long notes.")
        else:
            result['recommendations'].append("Standard model loaded. Consider training multi-duration model.")

        return jsonify(convert_to_serializable(result))

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Add this route after your other route definitions and before if __name__ == '__main__':

@app.route('/', methods=['GET'])
def index():
    """Root endpoint - provides API documentation"""
    try:
        model_loaded = model is not None
        extractor_loaded = extractor is not None
        model_type = metadata.get('model_type', 'Unknown') if metadata else 'Unknown'
        labels = metadata.get('labels', []) if metadata else []

        is_unified_model = 'Unified' in model_type or 'Multi-Duration' in model_type
        supports_short_notes = is_unified_model

        api_info = {
            'success': True,
            'message': 'Angklung CNN API Server',
            'version': 'Unified v1.0' if is_unified_model else 'Standard v1.0',
            'status': 'running',
            'model_info': {
                'type': str(model_type),
                'loaded': bool(model_loaded),
                'supports_short_notes': bool(supports_short_notes),
                'classes_supported': len(labels),
                'negative_detection': bool('no_angklung' in labels)
            },
            'available_endpoints': {
                'GET /': 'API documentation (this page)',
                'GET /health': 'Health check and detailed status',
                'GET /test_connection': 'Test model connection and capabilities',
                'GET /debug_connection': 'Debug connection issues',
                'GET /simple_test': 'Simple boolean test for Flutter',
                'GET /model_status': 'Detailed model status',
                'POST /predict_note': 'Upload audio file for note prediction'
            },
            'usage_examples': {
                'health_check': 'GET http://localhost:5000/health',
                'test_connection': 'GET http://localhost:5000/test_connection',
                'predict_note': 'POST http://localhost:5000/predict_note (with audio file in form-data)'
            }
        }

        if is_unified_model:
            api_info['unified_features'] = {
                'duration_detection': 'Automatically detects short vs long notes',
                'unified_processing': 'Handles all note durations in single model',
                'enhanced_accuracy': 'Improved accuracy for short notes'
            }

        return jsonify(convert_to_serializable(api_info))

    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': 'Angklung CNN API Server',
            'status': 'error',
            'error': str(e),
            'available_endpoints': [
                'GET /health',
                'GET /test_connection',
                'POST /predict_note'
            ]
        }), 500
@app.route('/health', methods=['GET'])
def health_check():
    try:
        model_loaded = model is not None
        extractor_loaded = extractor is not None
        model_type = metadata.get('model_type', 'Unknown') if metadata else 'Unknown'
        labels = metadata.get('labels', []) if metadata else []

        # Updated detection
        is_unified_model = 'Unified' in model_type or 'Multi-Duration' in model_type
        supports_short_notes = is_unified_model

        result = {
            'success': True,
            'server': f'Angklung {"Unified" if is_unified_model else "Standard"} CNN Server',
            'version': 'Unified v1.0' if is_unified_model else 'Standard v1.0',
            'status': 'healthy' if (model_loaded and extractor_loaded) else 'degraded',
            'components': {
                'cnn_model': 'loaded' if model_loaded else 'missing',
                'extractor': 'loaded' if extractor_loaded else 'missing',
                'label_encoder': 'loaded' if label_encoder is not None else 'missing',
                'metadata': 'loaded' if metadata else 'missing'
            },
            'model_info': {
                'type': str(model_type),
                'classes': len(labels),
                'negative_detection': bool('no_angklung' in labels),
                'supports_short_notes': bool(supports_short_notes)
            }
        }

        if is_unified_model:
            result['unified_status'] = {
                'duration_detection': 'ACTIVE',
                'duration_features': 'ACTIVE (11 features)',
                'unified_processing': 'ACTIVE - handles all durations automatically',
                'expected_accuracy': 0.88
            }

        result['technical_status'] = {
            'distance_normalization': 'ACTIVE',
            'enhanced_features': f'ACTIVE ({96 if is_unified_model else 85} features)',
            'cnn_architecture': 'ACTIVE',
            'distance_compensation': 'ACTIVE',
            'short_notes_support': 'ACTIVE' if supports_short_notes else 'INACTIVE'
        }

        return jsonify(convert_to_serializable(result))

    except Exception as e:
        return jsonify({
            'succq ess': False,
            'status': 'unhealthy',
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Get port from environment variable (Railway sets this to 8080)
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 for local development

    if load_multi_duration_model():  # Or rename to load_unified_model()
        print("Unified angklung model loaded successfully!")
        print("Model supports both short and long notes automatically!")
        print(f"Starting server on port {port}")
        app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        print("WARNING: No unified model files found, but starting server anyway for testing")
        print(f"Starting server on port {port}")
        app.run(host='0.0.0.0', port=port, debug=True, threaded=True)