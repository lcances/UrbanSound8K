# Unique augmentation to execute
unique_augments=dict(
    psc1    = "ubs8k.augmentation_utils.signal_augmentations.PitchShiftChoice(0.5, choice=(-3, -2, 2, 3))",
    PSC1    = "ubs8k.augmentation_utils.signal_augmentations.PitchShiftChoice(1.0, choice=(-3, -2, 2, 3))",
    psc2    = "ubs8k.augmentation_utils.signal_augmentations.PitchShiftChoice(0.5, choice=(-1.5, -1, 1, 1.5))",
    PSC2    = "ubs8k.augmentation_utils.signal_augmentations.PitchShiftChoice(1.0, choice=(-1.5, -1, 1, 1.5))",
    l1      = "ubs8k.augmentation_utils.signal_augmentations.Level(0.5, rate=(0.9, 1.1))",
    l2      = "ubs8k.augmentation_utils.signal_augmentations.Level(0.5, rate=(0.8, 1.2))",
    n1      = "ubs8k.augmentation_utils.signal_augmentations.Noise(0.5, target_snr=15)",
    n2      = "ubs8k.augmentation_utils.signal_augmentations.Noise(0.5, target_snr=20)",
    n3      = "ubs8k.augmentation_utils.signal_augmentations.Noise(0.5, target_snr=25)",
    rfd01   = "ubs8k.augmentation_utils.spec_augmentations.RandomFreqDropout(0.5, dropout=0.1)",
    rfd0075 = "ubs8k.augmentation_utils.spec_augmentations.RandomFreqDropout(0.5, dropout=0.075)",
    rfd02   = "ubs8k.augmentation_utils.spec_augmentations.RandomFreqDropout(0.5, dropout=0.2)",
    sn25    = "ubs8k.augmentation_utils.spec_augmentations.Noise(1.0, 25)",
    rfd005  = "ubs8k.augmentation_utils.spec_augmentations.RandomFreqDropout(0.5, dropout=0.05)",
)

reverse_unique_augment = dict(zip(unique_augments.values(), unique_augments.keys()))
