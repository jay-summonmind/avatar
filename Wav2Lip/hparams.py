from glob import glob
import os

def get_image_list(data_root, split):
    filelist = []

    with open(f'filelists/{split}.txt') as f:
        for line in f:
            line = line.strip()
            if ' ' in line:
                line = line.split()[0]
            filelist.append(os.path.join(data_root, line))

    return filelist


class HParams:
    def __init__(self, **kwargs):
        self.data = {}
        for key, value in kwargs.items():
            self.data[key] = value

    def __getattr__(self, key):
        if key not in self.data:
            raise AttributeError(f"'HParams' object has no attribute {key}")
        return self.data[key]

    def set_hparam(self, key, value):
        self.data[key] = value

    def values(self):
        return self.data


# 🔥 OOM-SAFE + LOW VRAM BEST PRACTICE (8GB GPU)
hparams = HParams(
    num_mels=80,

    # Audio preprocessing
    rescale=True,
    rescaling_max=0.9,
    mel_fmin=80.0,
    mel_fmax=7600.0,

    use_lws=False,
    n_fft=800,
    hop_size=200,
    win_size=200,
    sample_rate=16000,

    frame_shift_ms=None,

    # Normalization
    signal_normalization=True,
    allow_clipping_in_normalization=True,
    symmetric_mels=True,
    max_abs_value=4.0,

    preemphasize=True,
    preemphasis=0.97,

    # Limits
    min_level_db=-100,
    ref_level_db=20,
    fmin=55,
    fmax=7600,

    ###################### TRAINING (VRAM OPTIMIZED) ######################
    img_size=64,          # ↓ reduce from 96 to avoid OOM
    fps=25,

    batch_size=1,         # 🔥 MUST for 8GB VRAM (was missing = error)
    initial_learning_rate=1e-4,
    nepochs=200000,

    num_workers=0,        # 🔥 critical fix (16 → 0 to stop RAM explosion)
    checkpoint_interval=3000,
    eval_interval=3000,
    save_optimizer_state=True,

    # SyncNet (lightweight settings)
    syncnet_wt=0.0,
    syncnet_batch_size=8,     # reduced from 64 (huge memory saver)
    syncnet_lr=1e-4,
    syncnet_eval_interval=10000,
    syncnet_checkpoint_interval=10000,

    # Discriminator
    disc_wt=0.07,
    disc_initial_learning_rate=1e-4,
)


def hparams_debug_string():
    values = hparams.values()
    hp = [f"  {name}: {values[name]}" for name in sorted(values)]
    return "Hyperparameters:\n" + "\n".join(hp)