# ============================================================
# PAM / k-medoids clustering for the 20 scaled persona variables
#
# INPUT:
#   Selected_questions_variables_the20variables_scaled_01.xlsx
#
# OUTPUT:
#   1) PAM_clustering_results.xlsx
#      - Evaluation_Summary
#      - Assignments
#      - Cluster_Means_k2 ... Cluster_Means_k6
#      - Medoids
#
# PURPOSE:
#   - Run PAM / k-medoids for k = 2, 3, 4, 5, 6
#   - Compare solutions using silhouette score and cluster sizes
#   - Save assignments for later profiling and persona interpretation
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances, silhouette_score


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

input_file = Path("Selected_questions_variables_20260406_the20variables_scaled_01.xlsx")
output_file = Path("PAM_clustering_results_20260406.xlsx")


# ------------------------------------------------------------
# 2. Read scaled data
# ------------------------------------------------------------

df = pd.read_excel(input_file, sheet_name="Scaled_0_1")
df.columns = df.columns.astype(str).str.strip()

print("Input shape:", df.shape)
print("Columns:", df.columns.tolist())

# Keep a clean copy of the clustering matrix
X = df.copy()

# Safety: ensure numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Check for missing values
n_missing = int(X.isna().sum().sum())
if n_missing > 0:
    raise ValueError(f"Input data still contains {n_missing} missing values.")

# Convert to NumPy
X_np = X.to_numpy(dtype=float)

# Add respondent row id for later merging
df_assignments = pd.DataFrame({
    "Respondent_Row": np.arange(1, len(X) + 1)
})


# ------------------------------------------------------------
# 3. Distance matrix
# ------------------------------------------------------------

# Manhattan distance is a sensible choice for scaled survey data
D = pairwise_distances(X_np, metric="manhattan")

print("Distance matrix shape:", D.shape)


# ------------------------------------------------------------
# 4. PAM / k-medoids implementation
# ------------------------------------------------------------

def assign_points(distance_matrix: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    """
    Assign each point to the nearest medoid.
    """
    distances_to_medoids = distance_matrix[:, medoids]
    labels = np.argmin(distances_to_medoids, axis=1)
    return labels


def compute_total_cost(distance_matrix: np.ndarray, medoids: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute total clustering cost: sum of distances to assigned medoids.
    """
    cost = 0.0
    for cluster_id, medoid_idx in enumerate(medoids):
        cluster_points = np.where(labels == cluster_id)[0]
        cost += distance_matrix[cluster_points, medoid_idx].sum()
    return float(cost)


def update_medoids(distance_matrix: np.ndarray, labels: np.ndarray, current_medoids: np.ndarray) -> np.ndarray:
    """
    For each cluster, choose the point within that cluster that minimises
    the total intra-cluster distance.
    """
    new_medoids = current_medoids.copy()

    for cluster_id in range(len(current_medoids)):
        cluster_points = np.where(labels == cluster_id)[0]

        # If a cluster is empty, keep the current medoid
        if len(cluster_points) == 0:
            continue

        # Sub-distance matrix for points inside this cluster
        cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]

        # Sum distances from each candidate point to all others in the cluster
        total_distances = cluster_distances.sum(axis=1)

        # Best medoid is the point with minimum total distance
        best_local_index = np.argmin(total_distances)
        new_medoids[cluster_id] = cluster_points[best_local_index]

    return new_medoids


def pam_single_run(
    distance_matrix: np.ndarray,
    k: int,
    random_state: int = 42,
    max_iter: int = 200
):
    """
    Single PAM run with random medoid initialisation.
    """
    rng = np.random.default_rng(random_state)
    n = distance_matrix.shape[0]

    # Random initial medoids
    medoids = np.array(rng.choice(n, size=k, replace=False), dtype=int)

    for _ in range(max_iter):
        labels = assign_points(distance_matrix, medoids)
        new_medoids = update_medoids(distance_matrix, labels, medoids)

        # Stop if medoids no longer change
        if np.array_equal(new_medoids, medoids):
            break

        medoids = new_medoids

    labels = assign_points(distance_matrix, medoids)
    cost = compute_total_cost(distance_matrix, medoids, labels)

    return medoids, labels, cost


def pam_best_of_n(
    distance_matrix: np.ndarray,
    k: int,
    n_init: int = 30,
    base_random_state: int = 42,
    max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray, float]:
    best_cost = np.inf
    best_medoids: np.ndarray | None = None
    best_labels: np.ndarray | None = None

    for i in range(n_init):
        medoids, labels, cost = pam_single_run(
            distance_matrix=distance_matrix,
            k=k,
            random_state=base_random_state + i,
            max_iter=max_iter
        )

        if cost < best_cost:
            best_cost = float(cost)
            best_medoids = np.asarray(medoids, dtype=int)
            best_labels = np.asarray(labels, dtype=int)

    if best_medoids is None or best_labels is None:
        raise RuntimeError(f"PAM failed to produce a valid solution for k={k}")

    return best_medoids, best_labels, best_cost


# ------------------------------------------------------------
# 5. Run PAM for k = 2, 3, 4, 5, 6
# ------------------------------------------------------------

k_values = [2, 3, 4, 5, 6]

evaluation_rows = []
medoid_rows = []

all_cluster_means = {}
all_labels = {}

for k in k_values:
    print(f"\nRunning PAM for k = {k} ...")

    medoids, labels, cost = pam_best_of_n(
        distance_matrix=D,
        k=k,
        n_init=30,          # increase if you want a more exhaustive search
        base_random_state=42,
        max_iter=200
    )

    # Silhouette using precomputed distance matrix
    sil = silhouette_score(D, labels, metric="precomputed")

    # Cluster sizes
    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_size_map = {int(lbl) + 1: int(cnt) for lbl, cnt in zip(unique_labels, counts)}

    smallest_cluster = min(cluster_size_map.values())
    largest_cluster = max(cluster_size_map.values())

    # Store evaluation summary
    evaluation_rows.append({
        "k": k,
        "Average_Silhouette": round(float(sil), 4),
        "Total_Cost": round(float(cost), 4),
        "Smallest_Cluster_Size": smallest_cluster,
        "Largest_Cluster_Size": largest_cluster,
        "Cluster_Sizes": str(cluster_size_map)
    })

    # Store labels (1-based for readability)
    all_labels[k] = labels + 1
    df_assignments[f"Cluster_k{k}"] = labels + 1

    # Store medoids
    for cluster_id, medoid_idx in enumerate(medoids, start=1):
        medoid_rows.append({
            "k": k,
            "Cluster": cluster_id,
            "Medoid_Row": int(medoid_idx) + 1   # +1 to match Excel-style row numbering
        })

    # Cluster means on the 20 scaled variables
    df_with_labels = X.copy()
    df_with_labels["Cluster"] = labels + 1

    cluster_means = df_with_labels.groupby("Cluster").mean(numeric_only=True)
    cluster_means.insert(0, "Cluster_Size", df_with_labels["Cluster"].value_counts().sort_index())
    cluster_means = cluster_means.reset_index()

    all_cluster_means[k] = cluster_means

    print(f"k = {k} complete | silhouette = {sil:.4f} | sizes = {cluster_size_map}")


# ------------------------------------------------------------
# 6. Save outputs
# ------------------------------------------------------------

df_evaluation = pd.DataFrame(evaluation_rows)
df_medoids = pd.DataFrame(medoid_rows)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    # Summary sheet
    df_evaluation.to_excel(writer, sheet_name="Evaluation_Summary", index=False)

    # Assignments sheet
    df_assignments.to_excel(writer, sheet_name="Assignments", index=False)

    # Medoids sheet
    df_medoids.to_excel(writer, sheet_name="Medoids", index=False)

    # One cluster-means sheet per k
    for k, df_means in all_cluster_means.items():
        df_means.to_excel(writer, sheet_name=f"Cluster_Means_k{k}", index=False)

print(f"\nResults saved to: {output_file.resolve()}")