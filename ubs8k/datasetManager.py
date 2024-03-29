import os
from typing import Union

from numpy.core._multiarray_umath import ndarray

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np

import librosa
import tqdm
import time
import h5py
import pandas as pd
import logging
from multiprocessing import Process, Manager


def conditional_cache_v2(func):
    def decorator(*args, **kwargs):
        key = kwargs.get("key", None)
        cached = kwargs.get("cached", None)
        
        if cached is not None and key is not None:
            if key not in decorator.cache.keys():
                decorator.cache[key] = func(*args, **kwargs)

            return decorator.cache[key]
        
        return func(*args, **kwargs)

    decorator.cache = dict()

    return decorator



class DatasetManager:
    class_correspondance = {"Air_conditioner": 0, "car_horn": 1, "Children_laying": 2,
                            "dog_bark": 3, "drilling": 4,
                            "engine_iddling": 5, "gun_shot": 6,
                            "jackhammer": 7, "siren": 8,
                            "street_music": 9}

    class_correspondance_reverse = dict(
        zip(
            class_correspondance.values(),
            class_correspondance.keys()
        )
    )
    NB_CLASS = 10
    LENGTH = 4

    def __init__(self, metadata_root, audio_root, sr: int = 22050, augments: tuple = (),
                 folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                 subsampling: float = 1.0, subsampling_method: str = "balance",
                 verbose=1):
        """

        :param metadata_root: base for metadata files
        :param audio_root: base for audio files
        :param sr: sampling rate
        :param augments: augmentation to apply (must be from signal_augmentation, spec_augmentation, img_augmentations)
        :param train_fold: can be empty
        :param val_fold: can be empty
        :param subsampling: percentage of the dataset to load (0 < subsampling <= 1.0)
        :param subsampling_method: [random | balance]
        :param verbose: 1 terminal, 2 notebooks
        """
        self.sr = sr
        self.metadata_root = metadata_root
        self.audio_root = audio_root
        self.folds = folds
        self.subsampling = subsampling
        self.subsampling_method = subsampling_method
        self.hdf_file = "urbansound8k_%s.hdf5" % self.sr
        self.augments = augments
        self.feat_val = None
        self.y_val = None

        # verbose mode
        self.verbose = verbose
        if self.verbose == 1: self.tqdm_func = tqdm.tqdm
        elif self.verbose == 2: self.tqdm_func = tqdm.tqdm_notebook

        # Store the dataset metadata information
        self.meta = {}
        self._load_metadata()

        # Store the raw audio
        self.audio: dict = self._hdf_to_dict(os.path.join(audio_root, self.hdf_file), folds)

        # while using subsampling, the metadata still contain the complete dataset. Since the
        # CoTrainingDataset uses this metadata to pick the file to load, it needs to be curated
        self._clean_metadata()

    @property
    def validation(self):
        raise NotImplementedError()

    def _load_metadata(self):
        metadata_path = os.path.join(self.metadata_root, "UrbanSound8K.csv")

        self.meta = pd.read_csv(metadata_path, sep=",")
        self.meta = self.meta.set_index("slice_file_name")
        self.meta["idx"] = list(range(len(self.meta)))

    def _subsample(self, filenames, fold, seed: int = 1234) -> list:
        """
        Select a fraction of the files following using the specified picking method.
        Since a fold of data is loaded a once, It is necessary to know on which fold we are working.

        :param filenames: The list of filenames inside the fold
        :param fold: The fold to work with
        :param seed: The seed for the random choice of the file. MANDATORY when using static augmentation.
        The _subsample function is called two times for each folds, yielding different split
        :return: A list of indexes
        """

        def random_pick() -> list:
            nb_file = len(filenames)
            idx = list(range(nb_file))

            # backup the previous random state (to not mess with other function using random generation)
            previous_state = np.random.get_state()

            np.random.seed(seed)
            rnd_idx = np.random.choice(idx, size=int(nb_file * self.subsampling), replace=False)

            np.random.set_state(previous_state)

            return rnd_idx

        def balanced_pick() -> list:
            nb_file = len(filenames)
            meta = self.meta.loc[self.meta.fold == fold]

            # order the dataframe following the filenames list
            meta = meta.reindex(index=filenames)

            # calc fold class distribution
            meta["distribution"] = [0] * len(meta)
            for c_idx in range(10):
                meta_class = meta.loc[meta.classID == c_idx]
                distribution = len(meta_class) / nb_file
                meta.loc[meta.classID == c_idx, "distribution"] = [distribution] * len(meta_class)
            meta["distribution"] /= sum(meta["distribution"])

            # sample the dataframe
            sample = meta.sample(frac=self.subsampling, weights="distribution", replace=False, random_state=seed)
            return sample.idx.values

        if self.subsampling_method == "random":
            return random_pick()
        elif self.subsampling_method == "balance":
            return balanced_pick()
        else:
            raise AttributeError("Subsampling method: %s does not exist" % self.subsampling_method)

    def _hdf_to_dict(self, hdf_path, folds: list, key: str = "data") -> dict:
        output = dict()

        # open hdf file
        with h5py.File(hdf_path, "r") as hdf:
            for fold in self.tqdm_func(folds):
                hdf_fold = hdf["fold%d" % fold]

                filenames = np.asarray(hdf_fold["filenames"])
                audios = np.asarray(hdf_fold[key])

                # Apply subsampling if needed
                selection_idx = list(range(len(filenames)))
                if self.subsampling != 1.0:
                    selection_idx = self._subsample(filenames, fold)

                fold_dict = dict(zip(filenames[selection_idx], audios[selection_idx]))
                output["fold%d" % fold] = fold_dict

        logging.info("nb file loaded: %d" % len(output))
        return output

    def _clean_metadata(self):
        final_filenames = []
        for fold_number in self.folds:
            final_filenames += list(self.audio["fold%d" % fold_number].keys())

        self.meta = self.meta.loc[self.meta.index.isin(final_filenames)]

    def load_audio(self, file_path):
        raw_data, sr = librosa.load(file_path, sr=self.sr, res_type="kaiser_fast")
        return raw_data, sr

    @conditional_cache_v2
    def extract_feature(self, raw_data, **kwargs):
        """
        extract the feature for the model. Cache behaviour is implemented with the two parameters filename and cached
        :param raw_data: to audio to transform
        :key key: Unique key link to the raw_data, used internally by the cache system
        :key cached: use or not the cache system
        :return: the feature extracted from the raw audio
        """
        feat = librosa.feature.melspectrogram(
            raw_data, self.sr, n_fft=2048, hop_length=512, n_mels=64, fmin=0, fmax=self.sr // 2)
        feat = librosa.power_to_db(feat, ref=np.max)
        
        return feat


class StaticManager(DatasetManager):

    def __init__(self, metadata_root, audio_root, sr: int = 22050,
                 static_augment_file: str = "", static_augment_list: tuple = (), augments: tuple = (),
                 folds: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
                 subsampling: float = 1.0, subsampling_method: str = "balance",
                 verbose=1):
        super().__init__(metadata_root, audio_root, sr, augments,
                         folds, subsampling, subsampling_method, verbose)

        self.static_augmentation_file = static_augment_file
        self.augment_list = static_augment_list
        self.hdf_path = static_augment_file

        self.static_augmentation = dict()

        # Pre-load all augmentation in augmentation list
        for augment in self.augment_list:
            self.add_augmentation(augment)

    def add_augmentation(self, augmentation_name):
        self.static_augmentation[augmentation_name] = self._hdfaug_to_dict(self.hdf_path, self.folds, augmentation_name)
        
    def list_augmentation_availables(self):
        with h5py.File(self.hdf_path, "r") as hdf:
            print(hdf["fold1"].keys())

    def _hdfaug_to_dict(self, hdf_path, folds: tuple, key: str = "data") -> dict:
        output = dict()

        # open hdf file
        with h5py.File(hdf_path, "r") as hdf:
            for fold in self.tqdm_func(folds):
                hdf_fold = hdf["fold%d" % fold]

                filenames = np.asarray(hdf_fold["filenames"])

                if key not in hdf_fold:
                    raise KeyError("augmentation %s doesn't exist. There is: %s" % (key, hdf_fold.keys()))
                audios = np.asarray(hdf_fold[key])

                # Apply subsampling if needed
                selection_idx = list(range(len(filenames)))
                if self.subsampling != 1.0:
                    selection_idx = self._subsample(filenames, fold)

                for idx in selection_idx:
                    filename = filenames[idx]
                    audio = audios[:, idx]
                    output[filename] = audio

        logging.info("nb file loaded: %d" % len(output))
        return output


if __name__ == '__main__':
    # audio_root = "../dataset/audio"
    # metadata_root = "../dataset/metadata"
    audio_root = os.path.join("E:/", "Corpus", "UrbanSound8K", "audio")
    metadata_root = os.path.join("E:/", "Corpus", "UrbanSound8K", "metadata")
    static_augment_file = os.path.join("E:/", "Corpus", "UrbanSound8K", "audio", "urbansound8k_22050_augmentations.hdf5")
    augment_list = ["I_PSC1"]




