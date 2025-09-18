
"""
Say you have 100 neurons with 1000 timpoints, distributed in 8 groups of 10 cells with -0,8 to 0,8 corrration in activity, Activity is Gcamp, variable in amplitude, (you can generate ground truth impulses, then convert them into exponetial rise and decay events). the last 20 neurons have random activity pattern. Generate this dataset, show me the sorted traces. provide python code, then we'll move to next steps of PCA analysis with cross-validated optimal number of components.

>> Here you go — synthetic 100×1000 GCaMP-like dataset with 8 correlated groups (10 cells each, target correlations sampled in [-0.8, 0.8]) plus 20 random neurons, then traces sorted by group & prototypicality and shown as a heatmap. The figure above is generated from the code below. Key variables exposed: time_axis, calcium_dff (100×1000), group_labels (len=100), and sort_index (the row order used for plotting).
"""

import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(123)

# ---------- sizes ----------
n_neurons = 100
t_points = 1000
dt = 0.1
time_axis = np.arange(t_points) * dt

# ---------- labels (8 groups of 10 + 20 random) ----------
group_labels = np.concatenate([np.full(10, g) for g in range(1, 9)] + [np.full(20, 9)])

# ---------- correlation targets per structured group ----------
group_corr_targets = {g: rng.uniform(-0.8, 0.8) for g in range(1, 9)}

# ---------- GCaMP-like double-exponential kernel ----------
tau_rise, tau_decay = 0.2, 1.2
kt = np.arange(0, int(5*tau_decay/dt)) * dt
kernel = (1 - np.exp(-kt/tau_rise)) * np.exp(-kt/tau_decay)
kernel /= kernel.max()

def convolve_events(spikes_bool, amps):
    impulse = np.zeros(spikes_bool.size, dtype=float)
    impulse[spikes_bool] = amps
    return np.convolve(impulse, kernel, mode='full')[:spikes_bool.size]

def safe_corr(x, y):
    sx, sy = x - x.mean(), y - y.mean()
    vx, vy = np.linalg.norm(sx), np.linalg.norm(sy)
    if vx < 1e-12 or vy < 1e-12: return 0.0
    return float(np.dot(sx, sy) / (vx * vy))

# ---------- generate fluorescence ----------
fluor = np.zeros((n_neurons, t_points))
idx = 0
for g in range(1, 9):
    # group latent bursts
    base_rate = 0.7  # Hz
    group_spikes = rng.random(t_points) < (base_rate*dt)
    burst_on = rng.random(t_points) < 0.015
    for b in np.where(burst_on)[0]:
        group_spikes[b:b+6] = True

    # opponent mask for anti-correlation (prefer when group is quiet)
    quiet_mask = ~group_spikes
    # optional dilation to strengthen complementarity
    from scipy.ndimage import binary_dilation
    quiet_mask = binary_dilation(quiet_mask, iterations=2)

    group_amps = np.exp(rng.normal(np.log(1.0), 0.4, size=group_spikes.sum()))
    group_latent = convolve_events(group_spikes, group_amps)

    c = group_corr_targets[g]
    mag = np.clip(abs(c), 1e-3, 0.99)

    for _ in range(10):
        if c >= 0:
            # positively correlated: shared latent + private events
            private = rng.random(t_points) < (0.25*dt)
            priv_amps = np.exp(rng.normal(np.log(0.7), 0.5, size=private.sum()))
            priv_latent = convolve_events(private, priv_amps)
            drive = np.sqrt(mag) * group_latent + np.sqrt(1-mag) * priv_latent
        else:
            # negatively correlated: events mostly when group is quiet
            opp = (rng.random(t_points) < (0.6*dt)) & quiet_mask
            opp |= (rng.random(t_points) < (0.05*dt))  # small random background
            opp_amps = np.exp(rng.normal(np.log(1.0), 0.5, size=opp.sum()))
            opp_latent = convolve_events(opp, opp_amps)
            drive = np.sqrt(mag) * opp_latent + np.sqrt(1-mag) * (0.3 * group_latent)

        gain = rng.uniform(0.9, 1.4)
        baseline = rng.uniform(0.4, 0.8)
        noise = rng.normal(0, 0.02, size=t_points)
        trace = baseline + gain * drive + noise
        trace = np.maximum(trace, 0.05)  # keep physical (non-negative)
        fluor[idx] = trace
        idx += 1

# random neurons (independent)
for _ in range(20):
    rate = rng.uniform(0.1, 1.0)
    spikes = rng.random(t_points) < (rate*dt)
    amps = np.exp(rng.normal(np.log(0.9), 0.5, size=spikes.sum()))
    latent = convolve_events(spikes, amps)
    gain = rng.uniform(0.7, 1.3)
    baseline = rng.uniform(0.3, 0.9)
    noise = rng.normal(0, 0.03, size=t_points)
    trace = np.maximum(baseline + gain*latent + noise, 0.05)
    fluor[idx] = trace
    idx += 1

# ---------- convert to dF/F ----------
calcium_dff = np.zeros_like(fluor)
for i in range(n_neurons):
    F0 = max(np.percentile(fluor[i], 20), 1e-3)
    calcium_dff[i] = (fluor[i] - F0) / F0

# ---------- sort by group then by correlation to group centroid ----------
sort_index = []
for g in range(1, 10):  # 1..8 groups + 9=“random”
    ii = np.where(group_labels == g)[0]
    if ii.size == 0: 
        continue
    block = calcium_dff[ii]
    centroid = block.mean(axis=0)
    cors = np.array([safe_corr(tr, centroid) for tr in block])
    sort_index.extend(ii[np.argsort(-cors)].tolist())

sort_index = np.array(sort_index)
dff_sorted = calcium_dff[sort_index]
labels_sorted = group_labels[sort_index]

# ---------- plot sorted traces (heatmap) ----------
plt.figure(figsize=(12, 6))
im = plt.imshow(dff_sorted, aspect='auto', interpolation='nearest')
plt.xlabel('Time (samples)')
plt.ylabel('Neurons (sorted by group & prototypicality)')
plt.title('Synthetic GCaMP dF/F — Sorted Traces')
cbar = plt.colorbar(im)
cbar.set_label('dF/F0')

row = 0
for g in range(1, 10):
    cnt = np.sum(labels_sorted == g)
    if cnt > 0:
        plt.hlines(row + cnt - 0.5, xmin=0, xmax=t_points-1, linewidth=1)
        row += cnt

plt.tight_layout()
plt.show()

"""
Now scramble calcium_dff in place data order
"""

scramble_idx = rng.permutation(n_neurons)
calcium_dff = calcium_dff[scramble_idx]

# Show scramble results
plt.figure(figsize=(12, 6))
im = plt.imshow(calcium_dff, aspect='auto', interpolation='nearest')
plt.xlabel('Time (samples)')
plt.ylabel('Neurons (scrambled)')
plt.title('Synthetic GCaMP dF/F — Scrambled Traces')
cbar = plt.colorbar(im)
cbar.set_label('dF/F0')
plt.tight_layout()
plt.show()

"""
excellent. Now, using best practies, do a PCA, and plot 3D PCA . Then show me how to get cross validated optimal number of component. i want to see the plot showing how we find the optimal number, and also a good explained variance plot, and run the code on the previous data. plot all the requested figure (cross validation perf vs N components, with some confidence bands) and then a 3D PCA of the individual neurons

>>Done. I:

1. fit PCA (features=time, observations=neurons) after z-scoring timepoints,
2. computed cross-validated held-out explained variance (CVEV) vs number of components with 95% CI bands and marked both the max-CVEV choice and the 1-SE parsimonious choice,
3. plotted cumulative explained variance on the full fit, and
4. showed a 3D scatter of neurons in PC1–PC3 colored by group.

The figures are displayed above, and the printed line reports:

* `k_max` (best mean CVEV),
* `k_1se` (smallest k within one standard error of the best),
* `k@90% variance` from the full-data fit.

If you want a different CV scheme (e.g., block-CV across time, or leave-one-group-out), say the word and I’ll adapt it.
"""

# PCA analysis with cross-validated optimal number of components
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

# Reuse variables created earlier:
X = calcium_dff  # shape (n_neurons, n_timepoints)
labels = group_labels

n_neurons, n_time = X.shape

# ---------- Best practices pre-processing for PCA ----------
# Center & scale features (timepoints) across neurons so components emphasize patterns over magnitude
scaler = StandardScaler(with_mean=True, with_std=True)
Xz = scaler.fit_transform(X)  # (n_neurons x n_time)

# ---------- Fit PCA on all neurons for visualization ----------
max_k = min(50, n_neurons)  # cap for stability
pca_full = PCA(n_components=max_k, svd_solver="full", random_state=42)
scores_full = pca_full.fit_transform(Xz)  # neuron scores (n_neurons x k)
cum_evr = np.cumsum(pca_full.explained_variance_ratio_)

# ---------- Cross-validated selection of k ----------
def cv_reconstruction_score(Xz, k_list, n_splits=5, seed=123):
    """
    Returns:
      k_list,
      mean_cvev (held-out explained variance, higher is better),
      std_cvev (across folds)
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_k_scores = {k: [] for k in k_list}
    per_k_nmse = {k: [] for k in k_list}

    for train_idx, test_idx in kf.split(Xz):
        X_train, X_test = Xz[train_idx], Xz[test_idx]
        # variance of test data for normalization (feature-wise)
        var_test = np.var(X_test, axis=0, ddof=1).sum()  # total variance
        for k in k_list:
            p = PCA(n_components=k, svd_solver="full", random_state=seed)
            p.fit(X_train)
            # Reconstruct held-out neurons
            X_test_proj = p.transform(X_test)
            X_test_recon = p.inverse_transform(X_test_proj)
            # Normalized MSE across all features
            mse = np.mean((X_test - X_test_recon)**2)
            nmse = mse / (np.var(X_test, ddof=1))  # normalize by total variance per element
            # Held-out explained variance (CVEV) = 1 - NMSE
            cvev = 1.0 - nmse
            per_k_scores[k].append(cvev)
            per_k_nmse[k].append(nmse)

    k_arr = np.array(k_list)
    mean_cvev = np.array([np.mean(per_k_scores[k]) for k in k_list])
    std_cvev = np.array([np.std(per_k_scores[k], ddof=1) for k in k_list])
    return k_arr, mean_cvev, std_cvev

k_list = list(range(1, max_k+1))
k_arr, mean_cvev, std_cvev = cv_reconstruction_score(Xz, k_list, n_splits=5, seed=123)

# Select k: max CVEV, and 1-SE rule (smallest k within 1 std of max)
k_max = k_arr[np.argmax(mean_cvev)]
best_cvev = mean_cvev.max()
one_se_threshold = best_cvev - std_cvev[np.argmax(mean_cvev)]
# smallest k with mean_cvev >= one_se_threshold
k_1se = k_arr[np.where(mean_cvev >= one_se_threshold)[0][0]]

# ---------- Plot: CV performance vs components with 95% CI ----------
plt.figure(figsize=(8.5, 5.5))
# 95% CI ~ 1.96 * std/sqrt(n_splits); we used std across folds so approximate
n_splits = 5
ci95 = 1.96 * std_cvev / np.sqrt(n_splits)
plt.plot(k_arr, mean_cvev, marker='o', linewidth=1)
plt.fill_between(k_arr, mean_cvev - ci95, mean_cvev + ci95, alpha=0.2, label='~95% CI')
plt.axvline(k_max, linestyle='--', linewidth=1, label=f'k_max={k_max}')
plt.axvline(k_1se, linestyle=':', linewidth=1, label=f'k_1se={k_1se}')
plt.xlabel('Number of components (k)')
plt.ylabel('Held-out explained variance (CVEV)')
plt.title('Cross-validated PCA performance vs k')
plt.legend()
plt.tight_layout()
plt.show()

# ---------- Plot: Cumulative explained variance on full data ----------
plt.figure(figsize=(8.5, 5.5))
plt.plot(np.arange(1, max_k+1), cum_evr[:max_k], marker='o', linewidth=1)
plt.axhline(0.9, linestyle='--', linewidth=1, label='90% variance')
k90 = int(np.searchsorted(cum_evr, 0.9) + 1)
plt.axvline(k90, linestyle='--', linewidth=1, label=f'k@90%={k90}')
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance ratio')
plt.title('PCA cumulative explained variance (full fit)')
plt.legend()
plt.tight_layout()
plt.show()

# ---------- 3D PCA scatter (neurons in PC space) ----------
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

pc3 = PCA(n_components=3, svd_solver="full", random_state=42).fit_transform(Xz)

plt.figure(figsize=(8, 7))
ax = plt.axes(projection='3d')
# color by group, map group 9 (random) to a smaller marker
for g in np.unique(labels):
    sel = labels == g
    ax.scatter(pc3[sel, 0], pc3[sel, 1], pc3[sel, 2],
               s=30 if g != 9 else 20,
               alpha=0.8,
               label=f'Group {g}')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title('Neurons projected onto first 3 PCs')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()
plt.show()

print(f"k_max (max CVEV) = {k_max}, k_1se (more parsimonious) = {k_1se}, k@90% variance = {k90}")


"""
ok, now write code that would use PHATE instead of PCA, and do a hdbscan clustering on top, and plot average cluster traces. assume all librairies are installed
"""

# --- PHATE + HDBSCAN on calcium_dff, then average cluster traces ---
import numpy as np
import matplotlib.pyplot as plt

# 1) PHATE embedding (observations=neurons, features=timepoints)
import phate
from hdbscan import HDBSCAN
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

X = calcium_dff  # shape (n_neurons, n_time)
n_neurons, n_time = X.shape

# Optional: standardize over time to emphasize patterns over amplitude
from sklearn.preprocessing import StandardScaler
Xz = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

# PHATE: n_components=3 since we want a 3D scatter; you can tweak k, t
ph = phate.PHATE(n_components=3, knn=15, t='auto', mds_solver='smacof', random_state=42, n_jobs=-1)
emb3 = ph.fit_transform(Xz)  # (n_neurons, 3)

# 2) HDBSCAN clustering on the PHATE space
# Tip: use a small min_cluster_size to capture groups; adjust min_samples for noise robustness.
clusterer = HDBSCAN(min_cluster_size=6, min_samples=3, cluster_selection_epsilon=0.0)
labels_hdb = clusterer.fit_predict(emb3)  # -1 is noise
unique_labels = np.unique(labels_hdb)

# 3) 3D scatter of neurons in PHATE space
plt.figure(figsize=(8, 7))
ax = plt.axes(projection='3d')

# Simple palette: integers map to Matplotlib tab10 cycle; noise -> gray
def color_for(lbl):
    if lbl == -1:
        return (0.6, 0.6, 0.6, 0.7)  # gray-ish
    # cycle through default prop cycle
    return plt.rcParams['axes.prop_cycle'].by_key()['color'][lbl % 10]

for lbl in unique_labels:
    sel = labels_hdb == lbl
    ax.scatter(emb3[sel, 0], emb3[sel, 1], emb3[sel, 2],
               s=35 if lbl != -1 else 20,
               alpha=0.85 if lbl != -1 else 0.5,
               label=f'Cluster {lbl}' if lbl != -1 else 'Noise',
               c=[color_for(lbl)])

ax.set_xlabel('PHATE 1')
ax.set_ylabel('PHATE 2')
ax.set_zlabel('PHATE 3')
ax.set_title('PHATE (3D) + HDBSCAN clustering')
ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.0))
plt.tight_layout()
plt.show()

# 4) Average cluster traces (mean ± SEM) in ΔF/F space
#    (skip label -1 if you don’t want to include noise)
clusters_to_plot = [l for l in unique_labels if l != -1]
if len(clusters_to_plot) == 0:
    clusters_to_plot = unique_labels.tolist()  # fall back to include noise

plt.figure(figsize=(11, 6))
for lbl in clusters_to_plot:
    sel = (labels_hdb == lbl)
    block = X[sel]  # raw dF/F (not z-scored) to keep amplitude meaning
    mu = np.nanmean(block, axis=0)
    sem = np.nanstd(block, axis=0, ddof=1) / np.sqrt(max(1, block.shape[0]))

    # Optional light smoothing for plotting only (doesn't change data)
    # You can comment this out if you prefer raw lines
    from scipy.ndimage import gaussian_filter1d
    mu_s = gaussian_filter1d(mu, sigma=1.0)
    sem_s = gaussian_filter1d(sem, sigma=1.0)

    col = color_for(lbl)
    plt.plot(time_axis, mu_s, label=f'Cluster {lbl} (n={sel.sum()})', linewidth=1.6, c=col)
    plt.fill_between(time_axis, mu_s - sem_s, mu_s + sem_s, alpha=0.2, edgecolor='none', facecolor=col)

plt.xlabel('Time')
plt.ylabel('ΔF/F0')
plt.title('Average cluster traces (mean ± SEM)')
plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
plt.tight_layout()
plt.show()

# Provide outputs for later steps
phate_embedding_3d = emb3
cluster_labels_hdb = labels_hdb

print("PHATE+HDBSCAN done.",
      f"Found clusters: {sorted([l for l in unique_labels if l!=-1])} (noise n={np.sum(labels_hdb==-1)})")


# Label the cluster back on the original trace, as a vertical color band on the side of the trace
import matplotlib as mpl
import matplotlib.pyplot as plt

# Reuse variables from previous code
# calcium_dff (n_neurons x n_time)
n_neurons, n_time = calcium_dff.shape
cluster_labels = cluster_labels_hdb  # from previous HDBSCAN step
unique_clusters = np.unique(cluster_labels)
# Sort neurons by cluster label, then by prototypicality within cluster
sort_idx = []
for cl in unique_clusters:
    sel = np.where(cluster_labels == cl)[0]
    if sel.size == 0:
        continue
    block = calcium_dff[sel]
    centroid = block.mean(axis=0)
    cors = np.array([np.corrcoef(tr, centroid)[0, 1] for tr in block])
    sort_idx.extend(sel[np.argsort(-cors)].tolist())
sort_idx = np.array(sort_idx)
dff_sorted = calcium_dff[sort_idx] 
clusters_sorted = cluster_labels[sort_idx]
# Color map for clusters
cmap = plt.get_cmap('tab10')
norm = mpl.colors.Normalize(vmin=unique_clusters.min(), vmax=unique_clusters.max())
cluster_colors = {cl: cmap(norm(cl)) for cl in unique_clusters}
# Plot
plt.figure(figsize=(12, 6))
im = plt.imshow(dff_sorted, aspect='auto', interpolation='nearest', cmap='viridis')
plt.xlabel('Time (samples)')
plt.ylabel('Neurons (sorted by cluster)')
plt.title('Calcium dF/F (sorted by cluster)')
plt.colorbar(im, label='dF/F')

# ===================== Rastermap analysis & plotting =====================
# Run Rastermap on the neuron x time matrix and visualize the embedding
# and sorted traces. Assumes `rastermap` is installed in the environment.
try:
    import numpy as np
    import matplotlib.pyplot as plt
    from rastermap import Rastermap

    # Ensure data is float32 as recommended by rastermap
    spks = calcium_dff.astype('float32', copy=False)

    # Fit Rastermap; for small n_neurons (~100) defaults are fast and robust
    rm = Rastermap(n_PCs=min(100, spks.shape[1]),
                   n_clusters=min(60, max(10, spks.shape[0] // 2)),
                   locality=0.9,
                   time_lag_window=0,
                   verbose=False)
    rm = rm.fit(spks)

    # Outputs
    isort = rm.isort  # index order for neurons (only for igood neurons)
    embedding = rm.embedding  # (n_good, 1)
    Xemb = getattr(rm, 'X_embedding', None)  # optional binned/normalized view
    igood = getattr(rm, 'igood', None)  # boolean or indices of good neurons

    # Map sort back to all neurons if igood is provided
    if igood is not None:
        # igood can be boolean mask or index array
        if igood.dtype == bool:
            good_idx = np.where(igood)[0]
        else:
            good_idx = np.array(igood, copy=False)
        full_sort = np.arange(spks.shape[0])
        full_sort[:good_idx.size] = good_idx[isort]
        # Place non-good neurons after, in their original order
        bad_idx = np.setdiff1d(np.arange(spks.shape[0]), good_idx, assume_unique=False)
        full_sort[good_idx.size:] = bad_idx
        sort_for_plot = full_sort
    else:
        sort_for_plot = isort

    # Plot 1: Rastermap X_embedding if available
    if Xemb is not None:
        plt.figure(figsize=(12, 5))
        plt.imshow(Xemb, vmin=0, vmax=np.percentile(Xemb, 99), cmap='gray_r', aspect='auto')
        plt.xlabel('Time (samples)')
        plt.ylabel('Binned neurons')
        plt.title('Rastermap — normalized/binned activity (X_embedding)')
        plt.colorbar(label='normalized activity')
        plt.tight_layout()
        plt.show()

    # Plot 2: dF/F sorted by Rastermap order
    plt.figure(figsize=(12, 6))
    plt.imshow(spks[sort_for_plot], aspect='auto', interpolation='nearest', cmap='viridis')
    plt.xlabel('Time (samples)')
    plt.ylabel('Neurons (Rastermap sort)')
    plt.title('Calcium dF/F — Rastermap sorted')
    plt.colorbar(label='dF/F0')
    plt.tight_layout()
    plt.show()

    # Quick 1D embedding scatter to visualize structure along embedding
    if embedding is not None and embedding.ndim == 2 and embedding.shape[1] >= 1:
        y = embedding[:, 0]
        x = np.arange(y.size)
        plt.figure(figsize=(8, 3))
        plt.scatter(x, y, s=10, alpha=0.7)
        plt.xlabel('Good neurons (after quality filter)')
        plt.ylabel('Embedding dim 1')
        plt.title('Rastermap 1D embedding of neurons')
        plt.tight_layout()
        plt.show()

    print('Rastermap done: sorted and plotted. good_n=', len(y) if 'y' in locals() else 'n/a')
except Exception as e:
    import traceback
    print('Error running Rastermap:', str(e))
    traceback.print_exc()