**<u>Calcium imaging and Data Analysis</u>**

# Preparing the environnement

The analysis will use python 3.9. It may or may not work on more recent version. An environment template (SummerSchool24 basis.yaml) is provided to simplify your analysis. import it in anaconda, then you will have to add a few extra librairies by hand :

conda install ipython
conda install -c conda-forge ipympl

> if still missing, install 
> cycler
> ipython_genutils
> pyparsing 




# Pre-process data

## Datasets

### Sources

2 two-photon datasets are available. They were collected with a 3D two photon acousto-optic based microscope, with Real Time movement correction enabling stable recordings (Griffiths, Valera et al, Nat. Neurosci. 2020)

We will start with the motor cortex dataset. If the code is completed, you can run the code on a second dataset to see how different it is

* **M1 population imaging data**. Soma were recorded in the primary motor and visual cortex. The animal was  free to run. 

* **Arboreal scanning data**. The entire dendritic tree of a L2-3 Pyramidal cell in the CFA (M1) was recording using arbitrarily oriented 3D imaging patches, forming a "ribbon"  that follows tightly the cell morphology. The recording were then reprocessed using the following pipeline (essentially, final post hoc motion correction, creation of a mask on the dendritic shaft, compression of the signal in the mask using median signal along the dendrite, to reduce contaminations). Several behaviours were tracked : running speed of the wheel , motion index on various regions of the animal when using camera tracking. 1 traces per 20 microns linear segments. 

  

### Format

Most of the data types can be loaded using Matlab or python. FYI, Here is how the data was exported from MATLAB

```matlab
%% For the M1 dataset

%% Extract and store extracted variables in a struct
mat_data.behaviour = interpolate_to(params.external_var.encoder.value, size(result,2)); % interpolate_to is a custom function that match the sampling of the behaviour data to the calcium imaging data.
mat_data.behaviour_time = params.external_var.encoder.time;
mat_data.result = result;
mat_data.tax = params.timescale;
coords = cell2mat(cellfun(@(x,y) nanmean([nanmean(x,2),nanmean(y,2)],2), h.start_pixels, h.stop_pixels, 'UniformOutput', false));

save('M1_data.mat', 'mat_data', '-v7')
Do not use '-v7.3 if you need to load it with scipy io'
%% For the arborealscan dataset

% For N neurons and T timepoints : 
% behaviour 	= 1 x T array OR a strcuture of M variables, each one being a 1 x T array
% result 		= N x T matrix
% tax 			= 1 x T array 
% coords 		= N x 3 matrix
% event_t 		= 1 x T2 array. T2 can be used to subsample the signal

%% Extract and store behaviours
behaviour = {};
for el = obj.behaviours.types
    [~, behaviours.(el{1})] = obj.get_activity_bout(el{1});
    behaviour.(el{1}) = behaviours.(el{1}){1};
end

%% Extract and store extracted variables
tax 			= obj.t;
[~, coord_idx] 	= unique(obj.ref.indices.valid_ROIs_list);
coords 			= obj.ref.simplified_tree{1}.table(coord_idx,3:5);
event_t 		= vertcat(obj.event.peak_time{:})';
result 			= obj.rescaled_traces';

save('tree dataset.mat','tax','result','coords','behaviour','event_t','-v7'); % Do not use '-v7.3 if you need to load it with scipy io'
```

This is how you should load it in python. it is different depending on the source, as one was originally a structure in matlab (it is like a dictionary), while the others were stored as separated variables.

```python
from scipy.io import loadmat

# Selector for choosing the dataset
dataset_choice = "M1_data"  # Change to "tree_dataset" to use the other dataset

if dataset_choice == "tree_dataset":
    # Load the .mat file for the tree dataset. these were variables directly stored in the .mat file
    mat_path = "tree dataset.mat"  # Update this path to the location of your .mat file
    mat_data = loadmat(mat_path)
    
    # Extract the variables
    behaviour = mat_data['behaviour'].squeeze()  # Squeeze to remove single-dimensional entries
    result = mat_data['result']
    tax = mat_data['tax']
    coords = mat_data['coords']

elif dataset_choice == "M1_data":
    # Load the .mat file for the M1 dataset. # these were variables stored in a structure in the .mat file
    mat_path = "M1_data.mat"  # Data exported from matlab
    mat_data = loadmat(mat_path)
    
    # Extract the variables
    mat_data = mat_data['mat_data']
    behaviour = mat_data['behaviour'][0, 0].squeeze()  # Indexing to access the nested structure and squeeze to remove single-dimensional entries
    result = mat_data['result'][0, 0]  # Indexing to access the nested structure
    tax = mat_data['tax'][0, 0].squeeze()  # Indexing to access the nested structure and squeeze to remove single-dimensional entries

# Get the dimensions
T, N = len(behaviour), result.shape[0]

# Ensure result shape is (Individuals, Timepoints)
result.shape
```



# Roadmap

1. **Load the data,** 
   1. Plot the RAW traces. 
   2. Remove the NaN if any
   3. Smooth the traces
   4. Normalize the traces. Plot them again
2. **Detect events** using find_peaks, get the values and times
   1. Plot the peak values
   2. Plot general statistics about the peaks values
3. **Run correlations** between signal and behaviour
   1. Plot a correlation matrix
   1. If you are interested, we can dig into clustering of correlation matrices
4. **Run PCA** on the dataset
   1. Run and plot PC1, PC2 and PC3
   2. Run another dimensionality reduction methods. I proposed phate but feel free to propose soemething else
   3. Find a graphical way to identify behavior on these charts
5. **Run clustering on these dimensionality reduction methods**
   1. Get optimal cluster number
   2. Try different clustering
   3. Plot the mean trace using each 
   4. Optional : Show what happens with different normalization, or without normalization, and how it changes the outcome of clustering
6. Try to see how much signal can be explained with **machine Learning**
   1. Chose a type of model. Defend your choice
   2. Plot the prediction score across behaviours
      1.  Compare with and without KFold, and with and without hyperparameter optimization
   3. Plot the predicted behavior on top of the ground truth
   4. Extract the weights of the different neurons and how much they contribute to the model

