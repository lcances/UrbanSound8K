import os
import time

os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np
import pandas as pd
import torch
import tqdm
import weakref
from collections.abc import Callable, Iterable

from ubs8k.datasetManager import DatasetManager, StaticManager
from augmentation_utils.augmentations import SignalAugmentation, SpecAugmentation

import logging


class Dataset(torch.utils.data.Dataset):
    def __init__(self, 
                 manager: (DatasetManager, StaticManager), folds: tuple = (),
                 augments=(), augment_S: bool = True, augment_U: bool = True,
                 static_augmentation = {},
                 augment_choser: Callable = lambda x: np.random.choice(x, size=1),
                 cached=False, ):
        super().__init__()

        self.manager = manager
        self.folds = folds
        
        # augmentation management
        self.augments = augments + tuple(static_augmentation.keys())
        self.augment_S = augment_S
        self.augment_U = augment_U
        self.static_augmentation = static_augmentation
        self.augment_choser = augment_choser
        self.applied_augmentation = self.reset_augmentation_flags()
        
        # cache management
        self.cached = cached
        self.check_cache()

        # Get only necessary audio
        self.x = weakref.WeakValueDictionary()
        for fold_number in self.folds:
            self.x.update(self.manager.audio["fold%d" % fold_number])

        # Get only necessary metadata
        meta = self.manager.meta
        self.y = meta.loc[meta.fold.isin(self.folds)]

        # varialbe
        self.filenames = list(self.x.keys())
        self.s_idx = []
        self.u_idx = []
        
        # alias for verbose mode
        self.tqdm_func = self.manager.tqdm_func
        
    def check_cache(self):
        for aug in self.augments:
            if isinstance(aug, SignalAugmentation):
                self.cached = False
                logging.info("Cache system deactivate due to usage of signal augmentation")

    def __len__(self):
        nb_file = len(self.filenames)
        return nb_file

    def __getitem__(self, index: int):
        return self._generate_data(index)

    def split_s_u(self, s_ratio: float):
        if s_ratio == 1.0:
            return list(range(len(self.y))), []
        
        self.y["idx"] = list(range(len(self.y)))

        for i in range(DatasetManager.NB_CLASS):
            class_meta = self.y.loc[self.y.classID == i]
            nb_sample_s = int(np.ceil(len(class_meta) * s_ratio))
            
            total_idx = class_meta.idx.values
            self.s_idx_ = class_meta.sample(n=nb_sample_s).idx.values
            self.u_idx_ = set(total_idx) - set(self.s_idx_)

            self.s_idx += list(self.s_idx_)
            self.u_idx += list(self.u_idx_)

        return self.s_idx, self.u_idx

    def _generate_data(self, index: int):
        # load the raw_audio
        filename = self.filenames[index]
        raw_audio = self.x[filename]

        # recover ground truth
        y = self.y.at[filename, "classID"]
        
        # check if augmentation should be applied
        apply_augmentation = self.augment_S if index in self.s_idx else self.augment_U

        # chose augmentation, if no return an empty list
        augment_fn = self.augment_choser(self.augments) if self.augments else []

        # Apply augmentation, only one that applies on the signal will be executed
        raw_audio, cache_id = self._apply_augmentation(raw_audio, augment_fn, filename, apply_augmentation)

        # extract feature and apply spec augmentation
        feat = self.manager.extract_feature(raw_audio, key=cache_id, cached=self.cached)
        feat, _ = self._apply_augmentation(feat, augment_fn, filename, apply_augmentation)
        y = np.asarray(y)
        
        # call end of generation callbacks
        self.end_of_generation_callback()

        return feat, y
    
    def end_of_generation_callback(self):
        self.reset_augmentation_flags()

    def _pad_and_crop(self, raw_audio):
        LENGTH = DatasetManager.LENGTH
        SR = self.manager.sr

        if len(raw_audio) < LENGTH * SR:
            missing = (LENGTH * SR) - len(raw_audio)
            raw_audio = np.concatenate((raw_audio, [0] * missing))

        if len(raw_audio) > LENGTH * SR:
            raw_audio = raw_audio[:LENGTH * SR]

        return raw_audio
    
    # =================================================================================================================
    #   Augmentation
    # =================================================================================================================

    def _apply_augmentation(self, data, augment_list, filename: str = None, apply: bool = True):
        """
        Choose the proper augmentation function depending on the type. If augmentation_style is static and augType is
        SignalAugmentation, then call the static augmentation function, otherwise call the dynamic augmentation
        function.

        :param data: the data to augment.
        :param augment_fn: The taugmentation function to apply.
        :param filename: The filename of the current file to processe (usefull for static augmentation)
        :return: the augmented data.
        """
        # format augmentation
        if not isinstance(augment_list, Iterable):
            augment_list = [augment_list]
            
        # Apply all the augmentation inside the list
        augmented = data
        cache_id = filename
        
        # If no augmentation should be applied (see self.augment_S and self.augment_U)
        if not apply:
            return augmented, cache_id
        
        for augment_fn in augment_list:
            # If augmentaiton already applied, don't do it again
            if self.applied_augmentation.get(augment_fn, False):
                continue
                
            # dynamic signal augmentation
            if isinstance(augment_fn, SignalAugmentation):
                augmented = self._apply_dynamic_augmentation_helper(augment_fn, data)
                
                # Some function like TimeStretch modifiy the size of the signal.
                augmented = self._pad_and_crop(augmented)
                cache_id = None      # Dynamic augmentation can't be cached.

            # dynamic spec augmentation
            elif isinstance(augment_fn, SpecAugmentation):
                # Can apply spec augmentation only on spectrogram
                if len(data.shape) == 2:
                    augmented = self._apply_dynamic_augmentation_helper(augment_fn, data)
                    cache_id = None     #  SpecAugmentation happen after the cache system
                
            # Static augmentation
            elif isinstance(augment_fn, str):
                augmented, augment_str, flavor = self._apply_static_augmentation_helper(augment_fn, data, filename)

                if augment_str is None or flavor is None:
                    cache_id = filename
                else:
                    cache_id = "{}.{}.{}".format(filename, augment_str, flavor)

            # Unknow type, must be callable and can't be cached
            elif callable(augment_fn):
                augmented = augment_fn(data)
                
                # Just in case
                augmented = self._pad_and_crop(augmented)

                cache_id = None

            # unknown type and not callable ERROR
            else:
                raise TypeError("Augmentation must be callable. %s is not" % augment_fn)
                
        return augmented, cache_id

    def _apply_dynamic_augmentation_helper(self, augment_func, data):
        # Mark the augmentation as processed to avoid double application
        self.applied_augmentation[augment_func] = True
        
        return augment_func(data)

    def _apply_static_augmentation_helper(self, augment_str, data, filename):
        # Mark the augmentation as processed to avoid double application
        self.applied_augmentation[augment_func] = True
        
        apply = np.random.random()

        if apply <= self.static_augmentation.get(augment_str, 0.5):
            number_of_flavor = self.manager.static_augmentation[augment_str][filename].shape[0]
            flavor_to_use = np.random.randint(0, number_of_flavor)

            return self.manager.static_augmentation[augment_str][filename][flavor_to_use], augment_str, flavor_to_use
        return data, None, None
    
    def reset_augmentation_flags(self):
        self.applied_augmentation = dict(zip(self.augments, (False, ) * len(self.augments)))
        return self.applied_augmentation

    def set_static_augment_ratio(self, ratios: dict):
        self.static_augmentation_ratios = ratios


class CoTrainingDataset(torch.utils.data.Dataset):
    """Must be used with the CoTrainingSampler"""
    def __init__(self, manager: (DatasetManager, StaticManager), ratio: float = 0.1,
                 folds: tuple = (),
            unlabel_target: bool = False, static_augmentation: dict = {},
            augments: tuple = (), S_augment: bool = True, U_augment: bool = True,
            cached:bool = False):
        """
        Args:
            manager:
            unlabel_target (bool): If the unlabel target should be return or not
            augments (list):
        """
        super(CoTrainingDataset, self).__init__()

        self.manager = manager
        self.ratio = ratio
        self.folds = folds
        self.unlabel_target = unlabel_target

        self.augments = augments + tuple(static_augmentation.keys())
        self.static_augmentation = static_augmentation
        self.S_augment = S_augment
        self.U_augment = U_augment

        self.cached = cached
        if len(augments) != 0 and cached:
            logging.info("Cache system deactivate due to usage of online augmentation")
            self.cached = False

        # Get only necessary audio
        self.X = weakref.WeakValueDictionary()
        for fold_number in folds:
            self.X.update(self.manager.audio["fold%d" % fold_number])

        # Get only necessary metadata
        meta = self.manager.meta
        self.y = meta.loc[meta.fold.isin(self.folds)]

        self.y_S = pd.DataFrame()
        self.y_U = pd.DataFrame()

        self._prepare_cotraining_metadata()

        self.filenames_S = self.y_S.index.values
        self.filenames_U = self.y_U.index.values

        # alias for verbose mode
        self.tqdm_func = self.manager.tqdm_func

    def _prepare_cotraining_metadata(self):
        """Using the sampler nb of of supervised file, select balanced amount of
        file in each class
        """
        # Prepare ground truth, balanced class between S and U
        for i in range(DatasetManager.NB_CLASS):
            class_samples = self.y.loc[self.y.classID == i]

            nb_sample_S = int(np.ceil(len(class_samples) * self.ratio))

            if i == 0:
                self.y_S = class_samples[:nb_sample_S]
                self.y_U = class_samples[nb_sample_S:]

            else:
                class_meta_S = class_samples[:nb_sample_S]
                class_meta_U = class_samples[nb_sample_S:]
                self.y_S = pd.concat([self.y_S, class_meta_S])
                self.y_U = pd.concat([self.y_U, class_meta_U])

    def __len__(self) -> int:
        return len(self.filenames_S) + len(self.filenames_U)

    def __getitem__(self, batch_idx):
        if isinstance(batch_idx, (list, set, tuple)):
            return self._get_train(batch_idx)
        else:
            return self._get_val(batch_idx)

    def _get_val(self, idx):
        return self._generate_data(
            [idx],
            target_filenames=self.y.index.values,
            target_meta=self.y
        )

    def _get_train(self, batch_idx):
        views_indexes = batch_idx[:-1]
        U_indexes = batch_idx[-1]

        # Prepare views --------
        X, y = [], []
        for vi in views_indexes:
            X_V, y_V = self._generate_data(
                vi,
                target_filenames=self.filenames_S,
                target_meta=self.y_S,
                augment=self.S_augment
            )
            X.append(X_V)
            y.append(y_V)

        # Prepare U ---------
        target_meta = None if self.unlabel_target else self.y_U
        X_U, y_U = self._generate_data(
            U_indexes,
            target_filenames=self.filenames_U,
            target_meta=target_meta,
            augment=self.U_augment
        )
        X.append(X_U)
        y.append(y_U)

        return X, y

    def _generate_data(self, indexes: list, target_filenames: list, target_meta: pd.DataFrame = None, augment: bool = False):
        """
        Args:
            indexes (list):
            target_filenames (list):
            target_raw (dict):
            target_meta (pd.DataFrame):
        """
        # Get the corresponding filenames
        filenames = [target_filenames[i] for i in indexes]

        # Get the ground truth
        targets = 0
        if target_meta is not None:
            targets = [target_meta.at[name, "classID"] for name in filenames]

        # Get the raw_audio
        raw_audios = [self.X[name] for name in filenames]

        features = []
        for i, filename in enumerate(filenames):
            cache_id=None

            if augment:
                raw_audios[i], cache_id = self._apply_augmentation(raw_audios[i], SignalAugmentation, filename)
                
            raw_audios[i] = self._pad_and_crop(raw_audios[i])
            feat = self.manager.extract_feature(raw_audios[i], key=cache_id, cached=self.cached)

            if augment:
                feat, _ = self._apply_augmentation(feat, SpecAugmentation)

            features.append(feat)

        # Convert to np array
        return np.array(features), np.array(targets)

    def _apply_augmentation(self, data, augType, filename: str = None):
        """
        Choose the proper augmentation function depending on the type. If augmentation_style is static and augType is
        SignalAugmentation, then call the static augmentation function, otherwise call the dynamic augmentation
        function.

        It is possible to mix static and dynamic augmentation, static augmentation will be concider if the object in
        augment list is a string (not a callable)

        In case of static augmentation, the ratio must be define by the set_augment_ratio function, otherwise they are
        default to 0.5.

        :param data: the data to augment
        :param augType: The type augmentation( signal, spectrogram or image)
        :param filename: The filename of the current fiel to processe (usefull for static augmentation)
        :return: the augmented data
        """
        np.random.shuffle(self.augments)
        for augment in self.augments:
            # static augmentation are trigger on the signal phase and are represented by string
            if isinstance(augment, str) and augType == SignalAugmentation:
                augmented, augment_str, flavor = self._apply_static_augmentation_helper(augment, data, filename)

                if augment_str is None or flavor is None:
                    return augmented, filename

                cache_id = "{}.{}.{}".format(filename, augment_str, flavor)
                return augmented, cache_id

            else:
                if not isinstance(augment, str):
                    augmented = self._apply_dynamic_augmentation_helper(augment, data, augType)
                    return augmented, None   # Dynamic augmentation can't be cached

        return data, None

    def _apply_dynamic_augmentation_helper(self, augment_func, data, augType):
        if isinstance(augment_func, augType):
            return augment_func(data)

        return data

    def _apply_static_augmentation_helper(self, augment_str, data, filename):
        apply = np.random.random()

        if apply <= self.static_augmentation.get(augment_str, 0.5):
            number_of_flavor = self.manager.static_augmentation[augment_str][filename].shape[0]
            flavor_to_use = np.random.randint(0, number_of_flavor)

            return self.manager.static_augmentation[augment_str][filename][flavor_to_use], augment_str, flavor_to_use
        return data, None, None

    def _pad_and_crop(self, raw_audio):
        LENGTH = DatasetManager.LENGTH
        SR = self.manager.sr

        if len(raw_audio) < LENGTH * SR:
            missing = (LENGTH * SR) - len(raw_audio)
            raw_audio = np.concatenate((raw_audio, [0] * missing))

        if len(raw_audio) > LENGTH * SR:
            raw_audio = raw_audio[:LENGTH * SR]

        return raw_audio


if __name__ == '__main__':

    # load the data
    audio_root = "../dataset/audio"
    metadata_root = "../dataset/metadata"

    manager = StaticManager(
        metadata_root, audio_root,
        static_augment_file="../dataset/audio/urbansound8k_22050_augmentations.hdf5",
        static_augment_list=("N", ),
        folds=(1, 2, 3),
        subsampling=1.0
    )

    train_dataset = CoTrainingDataset(manager, folds=(1, 2), ratio=1.0, static_augmentation={"N": 1.0}, cached=True)
    val_dataset = CoTrainingDataset(manager, folds=(3, ), ratio=1.0, static_augmentation={"N": 1.0}, cached=True)

    print("len train dataset: ", len(train_dataset))
    print("len val dataset: ", len(val_dataset))

    test = [train_dataset[[[i], [], []]] for i in tqdm.tqdm(range(1000))]
    test = [train_dataset[[[i], [], []]] for i in tqdm.tqdm(range(1000))]
    test = [train_dataset[[[i], [], []]] for i in tqdm.tqdm(range(1000))]
    test = [train_dataset[[[i], [], []]] for i in tqdm.tqdm(range(1000))]
    test = [train_dataset[[[i], [], []]] for i in tqdm.tqdm(range(1000))]
    test = [train_dataset[[[i], [], []]] for i in tqdm.tqdm(range(1000))]
    test = [train_dataset[[[i], [], []]] for i in tqdm.tqdm(range(1000))]
    test = [train_dataset[[[i], [], []]] for i in tqdm.tqdm(range(1000))]

