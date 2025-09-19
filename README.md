# SummerSchool Repository
This repository gathers material for the Summer School. It contains data sets, code examples, and project instructions used throughout the sessions to provide hands-on practice with analysis tools, from basic regression to advanced spike inference methods.

## Installation Instructions

### 0. VScode

Install Git, GitHub, link to your account

install anaconda, then in coda prompt :

```powershell
conda init powershell
conda init cmd.exe
```

We recommend creating two separate Python environments for the Summer School:

### 1. General Tools Environment (`SummerSchool`)

#### Create the environment
```powershell
conda create -n SummerSchool python=3.13.7
```

#### Activate the environment
```powershell
conda activate SummerSchool
```

#### Install conda packages
```powershell
conda install -y ipykernel numpy pandas matplotlib scipy openpyxl scikit-learn seaborn statsmodels
conda install -c conda-forge hdbscan "ipympl>=0.9" "ipywidgets>=8.1" "matplotlib>=3.8" s_gd2 graphtools umap-learn
```

#### Install pip-only packages
```powershell
pip install alphashape statannotations
```
#### Unsupported pip package if you have a recent distrib (3.13?)

pip install --no-deps phate scprep


---

### 2. Cascade Environment (`cascade`)

The `Spike Inference (Cascade).ipynb` notebook depends on the external [Cascade repository](https://github.com/HelmchenLabSoftware/Cascade).
Clone that repository and create a dedicated `cascade` environment before running the notebook.

Cascade is a deep-learning toolbox for spike inference from calcium imaging. It requires a dedicated environment for compatibility with TensorFlow 2.3.

#### Clone the Cascade repository
```powershell
git clone https://github.com/HelmchenLabSoftware/Cascade.git
cd Cascade
```

#### Create and activate the environment
```powershell
conda create -n cascade python=3.7 pip
conda activate cascade
python -m pip install --upgrade pip setuptools wheel
```

#### Install Cascade and dependencies (CPU)
you must be in the cascade folder to run this
```powershell
pip install -e .
pip install "tensorflow==2.3.0" "tensorflow-estimator==2.3.0" "tensorboard==2.3.0" \
			"numpy==1.18.5" "scipy==1.4.1" "h5py==2.10.0" "protobuf<3.20"
pip install jupyter ipykernel
python -m ipykernel install --user --name=cascade --display-name "Python (cascade)"
```

#### (Optional) GPU support
```powershell
pip install "tensorflow-gpu==2.3.0"
# Make sure CUDA and cuDNN are compatible with TF 2.3
```

#### Sanity check
```powershell
python -c "import cascade2p, tensorflow as tf; print('cascade2p ->', cascade2p.__file__); print('TF ->', tf.__version__)"
```

#### Run the demo notebook
```powershell
jupyter notebook
# Open Demo scripts/Calibrated_spike_inference_with_Cascade.ipynb and select the 'Python (cascade)' kernel
```

Alternatively, use the [official Colab notebook](https://colab.research.google.com/github/HelmchenLabSoftware/Cascade/blob/master/Demo%20scripts/Calibrated_spike_inference_with_Cascade.ipynb) for a no-install trial.

For quick-start tips, see `Cascade Tuto.md`.

## Jupyter notebooks
The notebooks are organised roughly in the order they are used during the summer school. They can be run locally once the
environment above is installed, or explored directly in the `Projects/` and `Notes/` folders for supplemental material.

### Quick Guides

- **HandsOnGeneralStatistics.ipynb** – warm-up exercises covering descriptive statistics and hypothesis testing.
- **HandsOnPCAClust.ipynb** – guided practice on PCA and clustering using curated behavioural data sets.
- **HandsOnML.ipynb** – end-to-end introduction to classical machine-learning workflows (train/test split, pipelines, model evaluation).

### Advanced tips
- **Linear Regression M1.ipynb** – walkthrough of linear regression applied to motor cortex recordings.
- **Dimensionality Reduction and Clustering.ipynb** – extended notebook with additional PCA visualisations and clustering techniques.

### Other
- **Spike Inference (Cascade).ipynb** – demonstration of spike inference using the Cascade deep-learning toolbox (requires the separate `cascade` environment).

## MATLAB scripts and data
For those working in MATLAB, the following scripts reproduce the demonstrations from the notebooks. The accompanying `.mat` files hold the neural recordings and example data sets referenced throughout the school.

- **demo_PCA_time.m** / **demo_PCA_time.mlx** – MATLAB implementation of PCA over time.
- **extract_obj_for_summer_school.m** and **traces_for_cascade.m** – helper functions for data preparation.
- **M1_dendritic_tree_data.mat**, **M1_population_data.mat**, and **fish dataset.mat** – data sets used in the notebooks.

## Directories
- **AI_for_programming/** – resources on using AI as a programming assistant, including example prompts and slides.
- **Projects/** – instructions and project synopsis for the 2024 group projects.
- **Notes/** – additional notes such as the Jupyter extension setup guide.

## Miscellaneous
- **Program 2025.md** – full schedule of the school.
- Images and other supporting files used in the notebooks and slides.
- `traces_for_cascade.m` – MATLAB script to prepare calcium traces for Cascade.
- Other spike inference methods of interest include [MLspike](https://github.com/MLspike) and [OASIS](https://github.com/j-friedrich/OASIS).

## Demo_script_PCA_time

`Demo_script_PCA_time` visualises neuronal population activity in a reduced PCA space. The function accepts neural activity (`result`), behavioural data (`behaviour`), and optional event indices (`event_t`). If any inputs are omitted, the function automatically loads the demo data stored in `M1_population_data.mat`.

### Usage

```matlab
% Use bundled demo data
Demo_script_PCA_time();

% Custom data with a 5 point smoothing window and shorter fading memory
Demo_script_PCA_time(result, behaviour, event_t, 5, 100, [0.1 1]);
```
