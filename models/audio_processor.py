import numpy as np
import librosa

class AudioProcessor:
    def __init__(self, sr, segment_len, n_mels, n_fft):
        self.sr = sr
        self.segment_len = segment_len
        self.n_mels = n_mels
        self.n_fft = n_fft

    def wav_to_mel_spectrograms(self, wav_path):
        signal, sr = librosa.load(wav_path, sr=self.sr)
        segments = self._segment_audio(signal)
        return self._create_mel_spectrograms(segments)

    def _segment_audio(self, signal):
        segments = []
        step = self.segment_len
        for start in range(0, len(signal), step):
            segment = signal[start:start + self.segment_len]
            if len(segment) == self.segment_len:
                segment = self._normalize_segment(segment)
                segments.append(segment)
        while len(segments) < 10:
            segments.append(np.zeros(self.segment_len))
        return segments[:10]

    def _normalize_segment(self, segment):
        segment_mean = np.mean(segment)
        segment_std = np.std(segment)
        return (segment - segment_mean) / segment_std if segment_std != 0 else np.zeros_like(segment)

    def _create_mel_spectrograms(self, segments):
        mel_spectrograms = []
        for segment in segments:
            mel_spec = librosa.feature.melspectrogram(y=segment, sr=self.sr, n_mels=self.n_mels, n_fft=self.n_fft)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_db = self._pad_or_truncate(mel_spec_db)
            mel_spec_db = np.repeat(mel_spec_db[..., np.newaxis], 3, axis=-1)
            mel_spectrograms.append(mel_spec_db[np.newaxis, ...])
        return np.array(mel_spectrograms)

    def _pad_or_truncate(self, mel_spec_db):
        if mel_spec_db.shape[1] < 32:
            return np.pad(mel_spec_db, ((0, 0), (0, 32 - mel_spec_db.shape[1])), mode='constant')
        return mel_spec_db[:, :32]