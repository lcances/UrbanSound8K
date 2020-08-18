# UrbanSound8K
Urbansound8k dataset management

For personal use. You can use as it is.

dataset: `http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf`

# Dependancies & local installation
**Dependancies**
```bash
conda create -n ubs8k python=3 pip
conda activate ubs8k

conda install pytorch
conda install pandas
conda install numpy
conda install h5py
conda install pillow
conda install scikit-image
conda install tqdm

conda install -c conda-forge librosa
conda install numba=0.48.0

pip install --upgrade git+https://github.com/leocances/augmentation_utils.git
```

**local installation**
```bash
conda activate myenv
cd UrbanSound8k
pip install -e .
```

# Prepare the dataset
The system make use of HDF file to greatly reduce loading time.
- **-l**: Set the cropping & padding size of each file
- **-sr**: Set the sampling rate used to load the audio
- **-a**: Path to the audio directory (hdf file will be save here)
- **--compression**: All h5py compression are supported. 

The HDF file resulting contain one group corresponding for each folder.
Those groups contains one dataset for the raw_audio and a group for the list of
filename. The raw_audio and the file names are in the same order.

```bash
conda activate ubs8k
cd standalone
python mv_to_hdf.py -sr 22050 -l 4 -a ../dataset/audio
```

# Use the datasets
### without static augmentation
```python
from ubs8k.datasetManager import DatasetManager
from ubs8k.datasets import Dataset

# load the data
audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"

manager = DatasetManager(
    metadata_root, audio_root,
    folds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
)

train_dataset = Dataset(manager, folds=(1, 2, 3, 4, 5, 6, 7, 8, 9))
val_dataset = Dataset(manager, folds=(10, ))

print("len train dataset: ", len(train_dataset))
print("len val dataset: ", len(val_dataset))
```

### With static augmentation
Example when you want to use static augmentation for training as follow:
- A 75% chance of applying Noise
- A 25% chance of applying Pitch Shift

If for an augmentation several variation exist, then one is randomly chosen.

```python
from ubs8k.datasetManager import StaticManager
from ubs8k.datasets import Dataset

# load the data
audio_root = "../dataset/audio"
metadata_root = "../dataset/metadata"

manager = StaticManager(
    metadata_root, audio_root,
    folds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
    static_augment_file="../dataset/audio/urbansound8k_22050_augmentations.hdf5",
)
manager.add_augmentation("N")
manager.add_augmentation("PSC1")

train_dataset = Dataset(manager, folds=(1, 2, 3, 4, 5, 6, 7, 8, 9), 
    static_augmentation={"N": 0.75, "PSC1": 0.25}
)
val_dataset = Dataset(manager, folds=(10, ))

print("len train dataset: ", len(train_dataset))
print("len val dataset: ", len(val_dataset))
```

# pre-compute augmentation
The script **standalone/preprocess_augmentation.py** can use all the augmentation available in
augmentation_utils.signal_augmentations.

```bash
cd standalone
python preprocess_augmentations.py -sr 22050 \
    -l 4 \
    -a ../dataset/audio \
    -W 4 \
    -A="signal_augmentations.TimeStretch(1.0, rate=(0.8, 1.2))"  \
    -A="signal_augmentations.PitchShiftChoice(1.0, choice=(-1.5, -1, 1, 1.5) \
```
The example above will create (or update) the HDF file "urbansound8k_22050_augmentations.hdf5".

- This HDF file contain one group for each folder.
- For each folder and for each unique augmentation, a new dataset is created.
- If the augmentation already exist, then an new variation is created and the dataset updated.
The flavors are contain into an extra dimension of the dataset

####Example of HDF file architecture with several augmentation**

The dataset has a dimension of size 3 as follow: (nb_flavor, nb_files, nb_samples)

If each augmentation have 4 variations, then the dataset dimensions will be (4, nb_file, nb_samples)
```
urbansound8k_2200_augmentations.hdf5
|- fold1
   |- N
   |- PSC1
   |- PSC2
   |- TS
   |- filenames
|- fold2
   |- ...
...
```

#### How does the dataset are names
The name of the dataset on the HDF file are created based on the list of name found into
**ubs8k.augmentation_list.py**

If the argument **-A** is equal to one who is described in the list, then the corresponding name will be assigned to it.

If not, then a key is created using the initial of the class name. (*PitchShiftChoice --> PSC*). 
The key will be preceded by "I_" to mark this augmentation as non-consistent. Meaning that several parameters for this
augmentation could be mixed into the dataset.


<!--
## For my personnal use
For my personnal usage, workaround on CALMIP (limited user space and hardlink not working between different device)
 - 5Go is not enough to install everything at once.
 - It need some `conda clean --all` after installing big module (pytorch)
 - Best to have miniconda install in tmpdir directory
 - If not, have the venv directory inside the project and create a symlink
 `cd /miniconda/envs; ln -s /path/to/venv/ <name>`
 - Conda doesn't like symlink. use `CONDA_ALWAY_COPY=true` before calling conda
 - Pip cache is store under `~/.cache`
 
```Bash
CONDA_ALWAYS_COPY=true conda create -p /path/to/venv/ python=3 pip
cd ~/miniconda3/envs
ln -s /path/to/venv/ ubs8k
conda activate ubs8k

CONDA_ALWAYS_COPY=true conda install pytorch
conda clean --all
CONDA_ALWAYS_COPY=true conda install pandas numpy
...
```
-->
