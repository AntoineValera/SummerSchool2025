"""
Refactored pipeline to generate the synthetic calcium dataset and compare several
embedding + clustering methods with consistent preprocessing settings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold

import phate
import pacmap
from ivis import Ivis
from hdbscan import HDBSCAN
from scipy.ndimage import binary_dilation, gaussian_filter1d


@dataclass
class CalciumDataset:
    time_axis: np.ndarray
    dff: np.ndarray
    labels: np.ndarray
    sort_index: np.ndarray

    @property
    def sorted_dff(self) -> np.ndarray:
        return self.dff[self.sort_index]

    @property
    def sorted_labels(self) -> np.ndarray:
        return self.labels[self.sort_index]


def build_gcamp_kernel(dt: float, tau_rise: float, tau_decay: float) -> np.ndarray:
    kt = np.arange(0, int(5 * tau_decay / dt)) * dt
    kernel = (1 - np.exp(-kt / tau_rise)) * np.exp(-kt / tau_decay)
    peak = kernel.max()
    if peak > 0:
        kernel /= peak
    return kernel


def convolve_events(spikes_bool: np.ndarray, amps: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    impulse = np.zeros(spikes_bool.size, dtype=float)
    impulse[spikes_bool] = amps
    return np.convolve(impulse, kernel, mode="full")[: spikes_bool.size]


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    sx, sy = x - x.mean(), y - y.mean()
    vx, vy = np.linalg.norm(sx), np.linalg.norm(sy)
    if vx < 1e-12 or vy < 1e-12:
        return 0.0
    return float(np.dot(sx, sy) / (vx * vy))


def generate_synthetic_dataset(
    rng_seed: int = 123,
    n_groups: int = 8,
    neurons_per_group: int = 10,
    n_random: int = 20,
    t_points: int = 1000,
    dt: float = 0.1,
) -> CalciumDataset:
    rng = np.random.default_rng(rng_seed)
    n_neurons = n_groups * neurons_per_group + n_random

    time_axis = np.arange(t_points) * dt
    labels = np.concatenate(
        [np.full(neurons_per_group, g, dtype=int) for g in range(1, n_groups + 1)]
        + [np.full(n_random, 9, dtype=int)]
    )

    kernel = build_gcamp_kernel(dt=dt, tau_rise=0.2, tau_decay=1.2)
    fluor = np.zeros((n_neurons, t_points))
    idx = 0

    group_corr_targets = {g: rng.uniform(-0.8, 0.8) for g in range(1, n_groups + 1)}

    for g in range(1, n_groups + 1):
        base_rate = 0.7
        group_spikes = rng.random(t_points) < (base_rate * dt)
        burst_on = rng.random(t_points) < 0.015
        for b in np.where(burst_on)[0]:
            group_spikes[b : b + 6] = True

        quiet_mask = ~group_spikes
        quiet_mask = binary_dilation(quiet_mask, iterations=2)

        group_amps = np.exp(rng.normal(np.log(1.0), 0.4, size=group_spikes.sum()))
        group_latent = convolve_events(group_spikes, group_amps, kernel)

        target_corr = group_corr_targets[g]
        mag = np.clip(abs(target_corr), 1e-3, 0.99)

        for _ in range(neurons_per_group):
            if target_corr >= 0:
                private = rng.random(t_points) < (0.25 * dt)
                priv_amps = np.exp(rng.normal(np.log(0.7), 0.5, size=private.sum()))
                priv_latent = convolve_events(private, priv_amps, kernel)
                drive = np.sqrt(mag) * group_latent + np.sqrt(1 - mag) * priv_latent
            else:
                opp = (rng.random(t_points) < (0.6 * dt)) & quiet_mask
                opp |= rng.random(t_points) < (0.05 * dt)
                opp_amps = np.exp(rng.normal(np.log(1.0), 0.5, size=opp.sum()))
                opp_latent = convolve_events(opp, opp_amps, kernel)
                drive = np.sqrt(mag) * opp_latent + np.sqrt(1 - mag) * (0.3 * group_latent)

            gain = rng.uniform(0.9, 1.4)
            baseline = rng.uniform(0.4, 0.8)
            noise = rng.normal(0, 0.02, size=t_points)
            trace = baseline + gain * drive + noise
            fluor[idx] = np.maximum(trace, 0.05)
            idx += 1

    for _ in range(n_random):
        rate = rng.uniform(0.1, 1.0)
        spikes = rng.random(t_points) < (rate * dt)
        amps = np.exp(rng.normal(np.log(0.9), 0.5, size=spikes.sum()))
        latent = convolve_events(spikes, amps, kernel)
        gain = rng.uniform(0.7, 1.3)
        baseline = rng.uniform(0.3, 0.9)
        noise = rng.normal(0, 0.03, size=t_points)
        trace = baseline + gain * latent + noise
        fluor[idx] = np.maximum(trace, 0.05)
        idx += 1

    calcium_dff = np.zeros_like(fluor)
    for i in range(n_neurons):
        f0 = max(np.percentile(fluor[i], 20), 1e-3)
        calcium_dff[i] = (fluor[i] - f0) / f0

    accumulator = []
    for g in np.unique(labels):
        idxs = np.where(labels == g)[0]
        block = calcium_dff[idxs]
        centroid = block.mean(axis=0)
        cors = np.array([safe_corr(tr, centroid) for tr in block])
        accumulator.extend(idxs[np.argsort(-cors)].tolist())
    sort_index = np.array(accumulator, dtype=int)

    return CalciumDataset(time_axis=time_axis, dff=calcium_dff, labels=labels, sort_index=sort_index)


def standardize_time_features(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xz = scaler.fit_transform(X)
    return Xz, scaler


def build_group_palette(labels: np.ndarray) -> Dict[int, Tuple[float, float, float, float]]:
    unique = sorted(np.unique(labels))
    cmap = plt.get_cmap("tab10")
    palette: Dict[int, Tuple[float, float, float, float]] = {}
    for idx, lbl in enumerate(unique):
        if lbl == -1:
            palette[lbl] = (0.6, 0.6, 0.6, 0.7)
        else:
            palette[lbl] = cmap(idx % cmap.N)
    return palette


def build_cluster_palette(labels: np.ndarray) -> Dict[int, Tuple[float, float, float, float]]:
    unique = sorted(np.unique(labels), key=lambda x: (x == -1, x))
    cmap = plt.get_cmap("tab20")
    palette: Dict[int, Tuple[float, float, float, float]] = {}
    for idx, lbl in enumerate(unique):
        if lbl == -1:
            palette[lbl] = (0.6, 0.6, 0.6, 0.6)
        else:
            palette[lbl] = cmap(idx % cmap.N)
    return palette


def make_ivis_estimator(data: np.ndarray) -> Ivis:
    # Prefer Keras 3 API to avoid mixing tf.keras and keras
    try:
        from keras import layers, Model, Input  # Keras 3
    except Exception:
        # Fallback to tf.keras if needed
        import tensorflow as tf
        inputs = tf.keras.Input(shape=(data.shape[1],), dtype=tf.float32)
        x = tf.keras.layers.Dense(512, activation="elu")(inputs)
        x = tf.keras.layers.Dense(256, activation="elu")(x)
        base_network = tf.keras.Model(inputs, x, name="ivis_base")
    else:
        inputs = Input(shape=(data.shape[1],), dtype="float32")
        x = layers.Dense(512, activation="elu")(inputs)
        x = layers.Dense(256, activation="elu")(x)
        base_network = Model(inputs, x, name="ivis_base")

    ivis_estimator = Ivis(
        embedding_dims=3,
        verbose=True,
        epochs=400,
        n_epochs_without_progress=10,
        batch_size=16,
        k=10,
        model=base_network,  # already “called” via functional API
    )
    # Let the outer pipeline cast to float32 for Ivis
    ivis_estimator._requires_float32 = True  # marker used by run_embedding_with_clustering
    return ivis_estimator


def plot_sorted_traces(dataset: CalciumDataset) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(dataset.sorted_dff, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Neurons (sorted by group & prototypicality)")
    ax.set_title("Synthetic calcium dF/F sorted by group")
    fig.colorbar(im, ax=ax, label="dF/F")

    labels_sorted = dataset.sorted_labels
    row = 0
    for g in np.unique(labels_sorted):
        count = np.sum(labels_sorted == g)
        row += count
        ax.axhline(row - 0.5, color="white", linewidth=0.8)

    plt.tight_layout()
    plt.show()


def plot_mean_traces(dataset: CalciumDataset, palette: Dict[int, Tuple[float, float, float, float]]) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl in sorted(np.unique(dataset.labels)):
        mask = dataset.labels == lbl
        color = palette.get(lbl, (0.3, 0.3, 0.3, 0.8))
        ax.plot(dataset.time_axis, dataset.dff[mask].mean(axis=0), label=f"Group {lbl}", color=color)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean dF/F")
    ax.set_title("Mean calcium traces per group")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()

def plot_embedding_3d(
    embedding: np.ndarray,
    labels: np.ndarray,
    palette: Dict[int, Tuple[float, float, float, float]],
    title: str,
    label_formatter: Callable[[int], str] | None = None,
    legend: bool = True,
) -> None:
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    unique = sorted(np.unique(labels), key=lambda x: (x == -1, x))
    for lbl in unique:
        mask = labels == lbl
        if mask.sum() == 0:
            continue
        color = palette.get(lbl, (0.3, 0.3, 0.3, 0.8))
        size = 35 if lbl != 9 else 25
        alpha = 0.85 if lbl != -1 else 0.6
        label_text = label_formatter(lbl) if label_formatter else f"{lbl}"
        ax.scatter(
            embedding[mask, 0],
            embedding[mask, 1],
            embedding[mask, 2],
            s=size,
            color=color,
            alpha=alpha,
            label=label_text,
            edgecolor="none",
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.set_title(title)
    if legend:
        ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.show()


def cross_validated_explained_variance(
    X: np.ndarray,
    max_components: int | None = None,
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, np.ndarray]:
    if max_components is None:
        max_components = min(20, X.shape[0], X.shape[1])
    k_values = np.arange(1, max_components + 1)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    means = []
    sems = []
    fold_scores = []

    for k in k_values:
        scores = []
        for train_idx, test_idx in kf.split(X):
            model = PCA(n_components=k, svd_solver="full")
            model.fit(X[train_idx])
            proj = model.transform(X[test_idx])
            recon = model.inverse_transform(proj)
            residual = X[test_idx] - recon
            total = np.sum((X[test_idx] - X[test_idx].mean(axis=0)) ** 2)
            explained = 0.0 if total == 0 else 1.0 - np.sum(residual ** 2) / total
            scores.append(explained)
        scores = np.array(scores)
        fold_scores.append(scores)
        means.append(scores.mean())
        sems.append(scores.std(ddof=1) / np.sqrt(len(scores)))

    results = {
        "k_values": k_values,
        "cvev_mean": np.array(means),
        "cvev_sem": np.array(sems),
        "fold_scores": fold_scores,
    }

    best_idx = int(np.argmax(results["cvev_mean"]))
    results["k_max"] = int(k_values[best_idx])
    se = results["cvev_sem"][best_idx]
    within_se = results["cvev_mean"] >= results["cvev_mean"][best_idx] - se
    results["k_1se"] = int(k_values[np.where(within_se)[0][0]])
    return results


def plot_cross_validation(results: Dict[str, np.ndarray]) -> None:
    k_values = results["k_values"]
    mean = results["cvev_mean"]
    sem = results["cvev_sem"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(k_values, mean, marker="o")
    ax.fill_between(k_values, mean - 1.96 * sem, mean + 1.96 * sem, alpha=0.2)
    ax.axvline(results["k_max"], color="tab:green", linestyle="--", label=f"max @ {results['k_max']}")
    ax.axvline(results["k_1se"], color="tab:red", linestyle=":", label=f"1-SE @ {results['k_1se']}")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Held-out explained variance")
    ax.set_title("Cross-validated explained variance vs components")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_cumulative_variance(pca: PCA, k_at_90: int) -> None:
    cum = pca.explained_variance_ratio_.cumsum()
    ks = np.arange(1, cum.size + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, cum, marker="o")
    ax.axhline(0.9, color="tab:orange", linestyle="--", label="90% variance")
    ax.axvline(k_at_90, color="tab:purple", linestyle=":", label=f"k @ 90% = {k_at_90}")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_title("Cumulative explained variance (full PCA fit)")
    ax.legend()
    plt.tight_layout()
    plt.show()


def run_pca_workflow(Xz: np.ndarray, dataset: CalciumDataset, palette: Dict[int, Tuple[float, float, float, float]]) -> Dict[str, np.ndarray]:
    print("Running PCA workflow...")
    pca_three = PCA(n_components=3, svd_solver="full")
    embedding = pca_three.fit_transform(Xz)

    plot_embedding_3d(embedding, dataset.labels, palette, "PCA (3 components): groups")

    cv_results = cross_validated_explained_variance(Xz)
    plot_cross_validation(cv_results)

    full_components = min(Xz.shape)
    pca_full = PCA(n_components=full_components, svd_solver="full")
    pca_full.fit(Xz)
    k_at_90 = int(np.searchsorted(pca_full.explained_variance_ratio_.cumsum(), 0.9) + 1)
    plot_cumulative_variance(pca_full, k_at_90)

    summary = {
        "embedding": embedding,
        "cv_results": cv_results,
        "pca_full": pca_full,
        "k_at_90": k_at_90,
    }

    print(
        "PCA summary:",
        f"k_max={cv_results['k_max']}",
        f"k_1se={cv_results['k_1se']}",
        f"k@90%={k_at_90}",
    )
    return summary


def run_embedding_with_clustering(
    name: str,
    estimator_builder: Callable[[np.ndarray], object],
    Xz: np.ndarray,
    dataset: CalciumDataset,
    group_palette: Dict[int, Tuple[float, float, float, float]],
    cluster_kwargs: Dict[str, int],
) -> np.ndarray:
    print(f"Running {name} embedding...")
    estimator = estimator_builder(Xz)
    embedding_input = Xz.astype(np.float32, copy=False) if getattr(estimator, "_requires_float32", False) else Xz
    embedding = estimator.fit_transform(embedding_input)

    plot_embedding_3d(embedding, dataset.labels, group_palette, f"{name}: groups")

    clusterer = HDBSCAN(**cluster_kwargs)
    cluster_labels = clusterer.fit_predict(embedding)
    cluster_palette = build_cluster_palette(cluster_labels)

    plot_embedding_3d(
        embedding,
        cluster_labels,
        cluster_palette,
        f"{name}: HDBSCAN clusters",
        label_formatter=lambda c: "Noise" if c == -1 else f"Cluster {c}",
    )

    plot_cluster_traces(dataset.dff, cluster_labels, dataset.time_axis, name)
    plot_cluster_heatmap(dataset.dff, cluster_labels, name)

    print([{"label": int(lbl), "count": int((cluster_labels == lbl).sum())} for lbl in np.unique(cluster_labels)])
    return cluster_labels



def plot_cluster_traces(
    dff: np.ndarray,
    cluster_labels: np.ndarray,
    time_axis: np.ndarray,
    method_name: str,
    smoothing_sigma: float | None = 1.0,
) -> None:
    unique = sorted(np.unique(cluster_labels), key=lambda x: (x == -1, x))
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.get_cmap("tab10")

    for idx, lbl in enumerate(unique):
        mask = cluster_labels == lbl
        if mask.sum() == 0:
            continue
        block = dff[mask]
        mu = block.mean(axis=0)
        sem = block.std(axis=0, ddof=1) / np.sqrt(max(1, block.shape[0]))
        if smoothing_sigma is not None:
            mu = gaussian_filter1d(mu, sigma=smoothing_sigma)
            sem = gaussian_filter1d(sem, sigma=smoothing_sigma)
        color = (0.6, 0.6, 0.6) if lbl == -1 else cmap(idx % cmap.N)
        label = "Noise" if lbl == -1 else f"Cluster {lbl} (n={mask.sum()})"
        ax.plot(time_axis, mu, label=label, color=color, linewidth=1.6)
        ax.fill_between(time_axis, mu - sem, mu + sem, color=color, alpha=0.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dF/F")
    ax.set_title(f"{method_name}: mean trace per cluster")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()


def plot_cluster_heatmap(dff: np.ndarray, cluster_labels: np.ndarray, method_name: str) -> None:
    unique = sorted(np.unique(cluster_labels), key=lambda x: (x == -1, x))
    order = []
    for lbl in unique:
        idxs = np.where(cluster_labels == lbl)[0]
        order.extend(idxs.tolist())
    order = np.array(order, dtype=int)

    sorted_dff = dff[order]
    sorted_clusters = cluster_labels[order]

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(sorted_dff, aspect="auto", interpolation="nearest", cmap="viridis")
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Neurons (sorted by cluster)")
    ax.set_title(f"{method_name}: dF/F sorted by cluster")
    fig.colorbar(im, ax=ax, label="dF/F")

    row = 0
    for lbl in unique:
        count = np.sum(sorted_clusters == lbl)
        row += count
        ax.axhline(row - 0.5, color="white", linewidth=0.8)

    plt.tight_layout()
    plt.show()


def main() -> None:
    dataset = generate_synthetic_dataset()
    group_palette = build_group_palette(dataset.labels)

    plot_sorted_traces(dataset)
    plot_mean_traces(dataset, group_palette)

    Xz, _ = standardize_time_features(dataset.dff)

    run_pca_workflow(Xz, dataset, group_palette)

    embedding_specs = [
        (
            "PHATE",
            lambda data: phate.PHATE(
                n_components=3,
                knn=15,
                t="auto",
                mds_solver="smacof",
                random_state=42,
                n_jobs=-1,
            ),
        ),
        (
            "PaCMAP",
            lambda data: pacmap.PaCMAP(n_components=3, MN_ratio=2.0, FP_ratio=10.0, random_state=42),
        ),
        (
            "Ivis",
            make_ivis_estimator,
        ),
    ]

    cluster_kwargs = dict(min_cluster_size=3, min_samples=3)

    clusters = {}
    for name, builder in embedding_specs:
        cluster_labels = run_embedding_with_clustering(name, builder, Xz, dataset, group_palette, cluster_kwargs)
        clusters[name] = cluster_labels

    cluster_summary = {name: dict(zip(*np.unique(labels, return_counts=True))) for name, labels in clusters.items()}
    print('Cluster membership counts:', cluster_summary)


if __name__ == "__main__":
    main()
