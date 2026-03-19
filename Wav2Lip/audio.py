import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
from hparams import hparams as hp


# -------------------- AUDIO IO --------------------
def load_wav(path, sr):
    wav, _ = librosa.load(path, sr=sr)
    return wav


def save_wav(wav, path, sr):
    wav = wav / max(0.01, np.max(np.abs(wav)))  # safe normalization
    wav = (wav * 32767).astype(np.int16)
    wavfile.write(path, sr, wav)


# FIX: librosa.output.write_wav is deprecated
def save_wavenet_wav(wav, path, sr):
    wav = wav / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, (wav * 32767).astype(np.int16))


# -------------------- PREEMPHASIS --------------------
def preemphasis(wav, k, preemphasize=True):
    if preemphasize:
        return signal.lfilter([1, -k], [1], wav)
    return wav


def inv_preemphasis(wav, k, inv_preemphasize=True):
    if inv_preemphasize:
        return signal.lfilter([1], [1, -k], wav)
    return wav


# -------------------- HOP SIZE --------------------
def get_hop_size():
    if hp.hop_size is not None:
        return hp.hop_size
    assert hp.frame_shift_ms is not None
    return int(hp.frame_shift_ms / 1000 * hp.sample_rate)


# -------------------- SPECTROGRAMS --------------------
def linearspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _amp_to_db(np.abs(D)) - hp.ref_level_db
    return _normalize(S) if hp.signal_normalization else S


def melspectrogram(wav):
    D = _stft(preemphasis(wav, hp.preemphasis, hp.preemphasize))
    S = _linear_to_mel(np.abs(D))
    S = _amp_to_db(S) - hp.ref_level_db

    # 🔥 Critical: Remove NaNs (prevents inference crash)
    S = np.nan_to_num(S)

    return _normalize(S) if hp.signal_normalization else S


# -------------------- STFT --------------------
def _stft(y):
    return librosa.stft(
        y=y,
        n_fft=hp.n_fft,
        hop_length=get_hop_size(),
        win_length=hp.win_size,
        center=True  # IMPORTANT for frame alignment
    )


# -------------------- MEL BASIS --------------------
_mel_basis = None


def _linear_to_mel(spectrogram):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis()
    return np.dot(_mel_basis, spectrogram)


def _build_mel_basis():
    assert hp.mel_fmax <= hp.sample_rate // 2
    return librosa.filters.mel(
        sr=hp.sample_rate,
        n_fft=hp.n_fft,
        n_mels=hp.num_mels,
        fmin=hp.mel_fmin,
        fmax=hp.mel_fmax
    )


# -------------------- CONVERSIONS --------------------
def _amp_to_db(x):
    x = np.maximum(1e-5, x)  # prevent log(0)
    return 20 * np.log10(x)


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


# -------------------- NORMALIZATION --------------------
def _normalize(S):
    if hp.allow_clipping_in_normalization:
        if hp.symmetric_mels:
            return np.clip(
                (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value,
                -hp.max_abs_value,
                hp.max_abs_value,
            )
        else:
            return np.clip(
                hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db)),
                0,
                hp.max_abs_value,
            )

    if hp.symmetric_mels:
        return (2 * hp.max_abs_value) * ((S - hp.min_level_db) / (-hp.min_level_db)) - hp.max_abs_value
    else:
        return hp.max_abs_value * ((S - hp.min_level_db) / (-hp.min_level_db))


def _denormalize(D):
    if hp.symmetric_mels:
        return (((D + hp.max_abs_value) * -hp.min_level_db / (2 * hp.max_abs_value)) + hp.min_level_db)
    else:
        return ((D * -hp.min_level_db / hp.max_abs_value) + hp.min_level_db)