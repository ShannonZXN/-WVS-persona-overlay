# ============================================================
# PCA and t-SNE plots for clustering results
#
# PURPOSE:
#   1. Plot Test 1 clusters for k = 2..6 on PCA space
#   2. Plot Test 1 clusters for k = 2..6 on t-SNE space
#   3. Compare Test 1 vs Test 2 at k = 4 on PCA space
#   4. Compare Test 1 vs Test 2 at k = 4 on t-SNE space
#   5. Add transparent cluster regions using convex hulls
#
# INPUT FILES:
#   - Clustering results_1st set variables.xlsx
#   - Clustering results_2nd set variables.xlsx
#
# ASSUMPTION:
#   Both files contain an "Imputed Data" sheet with:
#   - feature columns
#   - Cluster_k2 ... Cluster_k6 columns
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

file_test1 = Path("Clustering results_1st set variables.xlsx")
file_test2 = Path("Clustering results_2nd set variables.xlsx")

# Output files
out_test1_pca = Path("test1_pca_k2_to_k6.png")
out_test1_tsne = Path("test1_tsne_k2_to_k6.png")
out_test1_vs_test2_pca = Path("test1_vs_test2_pca_k4.png")
out_test1_vs_test2_tsne = Path("test1_vs_test2_tsne_k4.png")


# ------------------------------------------------------------
# 2. Cluster colours
# ------------------------------------------------------------

cluster_color_map = {
    1: "tab:blue",
    2: "tab:orange",
    3: "tab:green",
    4: "tab:red",
    5: "tab:purple",
    6: "tab:brown"
}


# ------------------------------------------------------------
# 3. Helper for transparent cluster regions
# ------------------------------------------------------------

def add_cluster_hull(
    ax: Axes,
    points: np.ndarray,
    color,
    alpha: float = 0.10,
    linewidth: float = 1.5
) -> None:
    """
    Draw a transparent convex hull around a cluster if there are enough points.
    """
    if points.shape[0] < 3:
        return

    unique_points = np.unique(points, axis=0)
    if unique_points.shape[0] < 3:
        return

    try:
        hull = ConvexHull(unique_points)
        hull_points = unique_points[hull.vertices]

        polygon = Polygon(
            hull_points,
            closed=True,
            facecolor=color,
            edgecolor=color,
            alpha=alpha,
            linewidth=linewidth
        )
        ax.add_patch(polygon)
    except Exception:
        return


# ------------------------------------------------------------
# 4. Read data
# ------------------------------------------------------------

df1 = pd.read_excel(file_test1, sheet_name="Imputed Data")
df2 = pd.read_excel(file_test2, sheet_name="Imputed Data")

df1.columns = df1.columns.astype(str).str.strip()
df2.columns = df2.columns.astype(str).str.strip()

print("Test 1 shape:", df1.shape)
print("Test 2 shape:", df2.shape)


# ------------------------------------------------------------
# 5. Identify feature columns and cluster columns
# ------------------------------------------------------------

cluster_cols1 = [c for c in df1.columns if c.startswith("Cluster_k")]
cluster_cols2 = [c for c in df2.columns if c.startswith("Cluster_k")]

exclude_cols = {"Respondent_Row", "Cluster"}

feature_cols1 = [c for c in df1.columns if c not in set(cluster_cols1).union(exclude_cols)]
feature_cols2 = [c for c in df2.columns if c not in set(cluster_cols2).union(exclude_cols)]

print("Test 1 feature columns:", feature_cols1)
print("Test 2 feature columns:", feature_cols2)

# Convert features to numeric
for c in feature_cols1:
    df1[c] = pd.to_numeric(df1[c], errors="coerce")

for c in feature_cols2:
    df2[c] = pd.to_numeric(df2[c], errors="coerce")


# ------------------------------------------------------------
# 6. Keep complete rows needed for plotting
# ------------------------------------------------------------

needed_test1 = feature_cols1 + [c for c in ["Cluster_k2", "Cluster_k3", "Cluster_k4", "Cluster_k5", "Cluster_k6"] if c in df1.columns]
needed_test2 = feature_cols2 + [c for c in ["Cluster_k4"] if c in df2.columns]

df1_plot = df1.dropna(subset=needed_test1).copy()
df2_plot = df2.dropna(subset=needed_test2).copy()

X1 = df1_plot[feature_cols1].to_numpy(dtype=float)
X2 = df2_plot[feature_cols2].to_numpy(dtype=float)

print("Rows used in Test 1 plots:", X1.shape[0])
print("Rows used in Test 2 plots:", X2.shape[0])


# ------------------------------------------------------------
# 7. Embeddings
# ------------------------------------------------------------

# PCA for Test 1 (used consistently across k=2..6)
pca1 = PCA(n_components=2)
X1_pca = pca1.fit_transform(X1)
evr1 = pca1.explained_variance_ratio_

# PCA for Test 2 (used for comparison at k=4)
pca2 = PCA(n_components=2)
X2_pca = pca2.fit_transform(X2)
evr2 = pca2.explained_variance_ratio_

# To make t-SNE more stable/faster, reduce to 10 PCA dimensions first
pre_pca1 = PCA(n_components=min(10, X1.shape[1]))
X1_pre = pre_pca1.fit_transform(X1)

pre_pca2 = PCA(n_components=min(10, X2.shape[1]))
X2_pre = pre_pca2.fit_transform(X2)

tsne1 = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
    max_iter=500
)
X1_tsne = tsne1.fit_transform(X1_pre)

tsne2 = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42,
    max_iter=500
)
X2_tsne = tsne2.fit_transform(X2_pre)


# ------------------------------------------------------------
# 8. Helper function for k=2..6 panel plots
# ------------------------------------------------------------

def plot_embedding_panels(
    coords: np.ndarray,
    labels_df: pd.DataFrame,
    k_list: list[int],
    title: str,
    xlabel: str,
    ylabel: str,
    outpath: Path
) -> None:
    """
    Make a 2x3 panel plot for k=2..6 using the same embedding,
    with transparent convex-hull regions for clusters.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, k in enumerate(k_list):
        ax = axes[idx]
        col = f"Cluster_k{k}"

        labels = pd.to_numeric(labels_df[col], errors="coerce").astype(int).to_numpy()

        for cluster_id in sorted(np.unique(labels)):
            mask = labels == cluster_id
            cluster_points = coords[mask]
            color = cluster_color_map.get(cluster_id, None)

            add_cluster_hull(
                ax=ax,
                points=cluster_points,
                color=color,
                alpha=0.10,
                linewidth=1.5
            )

            ax.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                s=18,
                alpha=0.75,
                color=color,
                label=f"C{cluster_id}"
            )

        ax.set_title(f"k = {k}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)

    # Use the last panel for legend
    axes[5].axis("off")
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        axes[5].legend(handles, legend_labels, loc="center", frameon=True)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# 9. Plot Test 1: PCA for k=2..6
# ------------------------------------------------------------

plot_embedding_panels(
    coords=X1_pca,
    labels_df=df1_plot,
    k_list=[2, 3, 4, 5, 6],
    title=f"Test 1 clusters on PCA space (PC1={evr1[0]*100:.1f}%, PC2={evr1[1]*100:.1f}%)",
    xlabel="PC1",
    ylabel="PC2",
    outpath=out_test1_pca
)


# ------------------------------------------------------------
# 10. Plot Test 1: t-SNE for k=2..6
# ------------------------------------------------------------

plot_embedding_panels(
    coords=X1_tsne,
    labels_df=df1_plot,
    k_list=[2, 3, 4, 5, 6],
    title="Test 1 clusters on t-SNE space",
    xlabel="t-SNE 1",
    ylabel="t-SNE 2",
    outpath=out_test1_tsne
)


# ------------------------------------------------------------
# 11. Plot Test 1 vs Test 2: PCA at k=4
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, coords, dfp, title in [
    (axes[0], X1_pca, df1_plot, f"Test 1 (k = 4)\nPC1={evr1[0]*100:.1f}%, PC2={evr1[1]*100:.1f}%"),
    (axes[1], X2_pca, df2_plot, f"Test 2 (k = 4)\nPC1={evr2[0]*100:.1f}%, PC2={evr2[1]*100:.1f}%"),
]:
    labels = pd.to_numeric(dfp["Cluster_k4"], errors="coerce").astype(int).to_numpy()

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_points = coords[mask]
        color = cluster_color_map.get(cluster_id, None)

        add_cluster_hull(
            ax=ax,
            points=cluster_points,
            color=color,
            alpha=0.10,
            linewidth=1.5
        )

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=18,
            alpha=0.75,
            color=color,
            label=f"C{cluster_id}"
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=True)
fig.suptitle("PCA comparison: Test 1 vs Test 2 at k = 4", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out_test1_vs_test2_pca, dpi=300, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------
# 12. Plot Test 1 vs Test 2: t-SNE at k=4
# ------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, coords, dfp, title in [
    (axes[0], X1_tsne, df1_plot, "Test 1 (k = 4)"),
    (axes[1], X2_tsne, df2_plot, "Test 2 (k = 4)"),
]:
    labels = pd.to_numeric(dfp["Cluster_k4"], errors="coerce").astype(int).to_numpy()

    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        cluster_points = coords[mask]
        color = cluster_color_map.get(cluster_id, None)

        add_cluster_hull(
            ax=ax,
            points=cluster_points,
            color=color,
            alpha=0.10,
            linewidth=1.5
        )

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=18,
            alpha=0.75,
            color=color,
            label=f"C{cluster_id}"
        )

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=True)
fig.suptitle("t-SNE comparison: Test 1 vs Test 2 at k = 4", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out_test1_vs_test2_tsne, dpi=300, bbox_inches="tight")
plt.close(fig)


# ------------------------------------------------------------
# 13. Done
# ------------------------------------------------------------

print("Saved files:")
print(out_test1_pca.resolve())
print(out_test1_tsne.resolve())
print(out_test1_vs_test2_pca.resolve())
print(out_test1_vs_test2_tsne.resolve())
