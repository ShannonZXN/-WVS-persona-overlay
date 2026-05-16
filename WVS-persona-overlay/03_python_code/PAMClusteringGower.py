# ============================================================
# PAM / k-medoids clustering using Gower distance
# on pre-scaled, post-imputation 20-variable data
#
# INPUT:
#   1) Selected_questions_variables_20260331_the20variables.xlsx
#      (imputed, NOT min-max scaled yet)
#   2) Question and ratings_20260331.xlsx
#
# OUTPUT:
#   PAM_clustering_results_Gower_prescaled.xlsx
#
# PURPOSE:
#   - Reverse-code descending variables
#   - Compute Gower distance on original coded values
#   - Run PAM for k = 2, 3, 4, 5, 6
#   - Save evaluation summary and cluster outputs
# ============================================================

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score


# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

input_data_file = Path("Selected_questions_variables_20260406_the20variables.xlsx")
ratings_file = Path("Question and ratings_20260331.xlsx")
output_file = Path("PAM_clustering_results_Gower_prescaled_20260406.xlsx")


# ------------------------------------------------------------
# 2. Read files
# ------------------------------------------------------------

# Change sheet_name if needed
df_data = pd.read_excel(input_data_file, sheet_name="Imputed Data")
df_ratings = pd.read_excel(ratings_file, sheet_name=0)

df_data.columns = df_data.columns.astype(str).str.strip()
df_ratings.columns = df_ratings.columns.astype(str).str.strip()

print("Input data shape:", df_data.shape)
print("Ratings shape:", df_ratings.shape)
print("Data columns:", df_data.columns.tolist())


# ------------------------------------------------------------
# 3. Check required metadata columns
# ------------------------------------------------------------

required_cols = ["Question_Number", "Rating_Code", "Is_Error_Code", "Scale_Direction"]
missing_cols = [col for col in required_cols if col not in df_ratings.columns]
if missing_cols:
    raise ValueError(f"Missing required columns in ratings file: {missing_cols}")

df_ratings["Question_Number"] = df_ratings["Question_Number"].astype(str).str.strip()
df_ratings["Scale_Direction"] = (
    df_ratings["Scale_Direction"]
    .astype(str)
    .str.strip()
    .str.lower()
)
df_ratings["Rating_Code_Numeric"] = pd.to_numeric(df_ratings["Rating_Code"], errors="coerce")

# Convert Is_Error_Code to bool safely
df_ratings["Is_Error_Code"] = (
    df_ratings["Is_Error_Code"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"true": True, "false": False, "1": True, "0": False})
    .fillna(df_ratings["Is_Error_Code"])
)


# ------------------------------------------------------------
# 4. Keep only the 20 clustering variables
# ------------------------------------------------------------

questions = df_data.columns.tolist()
X = df_data.copy()

# Ensure numeric
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")

# Check missingness
n_missing = int(X.isna().sum().sum())
if n_missing > 0:
    raise ValueError(f"Input data still contains {n_missing} missing values.")


# ------------------------------------------------------------
# 5. Build valid-code ranges and reverse-code descending items
# ------------------------------------------------------------

valid_code_map = {}
direction_map = {}

for q in questions:
    sub = df_ratings[df_ratings["Question_Number"] == q].copy()

    valid_codes = (
        sub.loc[sub["Is_Error_Code"] == False, "Rating_Code_Numeric"]
        .dropna()
        .astype("float64")
        .to_numpy(dtype=float)
        .tolist()
    )
    valid_codes = sorted(valid_codes)

    if len(valid_codes) == 0:
        raise ValueError(f"No valid rating codes found for {q}")

    valid_code_map[q] = valid_codes

    direction_values = (
        sub["Scale_Direction"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    direction_values = [x for x in direction_values if x not in ["nan", "none", ""]]
    direction_map[q] = direction_values[0] if len(direction_values) > 0 else None

# Reverse-code descending variables
X_aligned = X.copy()

for q in questions:
    valid_codes = valid_code_map[q]
    min_code = min(valid_codes)
    max_code = max(valid_codes)
    direction = direction_map[q]

    if direction == "descending":
        # Reverse coding around min/max valid range
        X_aligned[q] = max_code + min_code - X_aligned[q]
        print(f"{q}: reverse-coded (descending -> ascending)")
    else:
        print(f"{q}: kept as-is")


# ------------------------------------------------------------
# 6. Compute Gower distance matrix
# ------------------------------------------------------------

def compute_gower_distance_numeric(df_numeric: pd.DataFrame, valid_code_map: dict[str, list[float]]) -> np.ndarray:
    """
    Compute Gower distance for numeric-coded questionnaire variables:
        d(i,j) = average over variables of |x_i - x_j| / range_k

    Uses valid code ranges from the metadata file.
    """
    X_np = df_numeric.to_numpy(dtype=float)
    n, p = X_np.shape
    D = np.zeros((n, n), dtype=float)

    # Range per variable from valid code metadata
    ranges = []
    for col in df_numeric.columns:
        valid_codes = valid_code_map[col]
        min_code = min(valid_codes)
        max_code = max(valid_codes)
        var_range = max_code - min_code
        if var_range == 0:
            var_range = 1.0  # safety
        ranges.append(var_range)

    ranges = np.asarray(ranges, dtype=float)

    # Pairwise Gower
    for i in range(n):
        diff = np.abs(X_np[i, :] - X_np) / ranges
        D[i, :] = diff.mean(axis=1)

    return D


D = compute_gower_distance_numeric(X_aligned, valid_code_map)

print("Distance matrix shape:", D.shape)
print("Distance matrix min/max:", float(D.min()), float(D.max()))


# ------------------------------------------------------------
# 7. PAM / k-medoids implementation
# ------------------------------------------------------------

def assign_points(distance_matrix: np.ndarray, medoids: np.ndarray) -> np.ndarray:
    distances_to_medoids = distance_matrix[:, medoids]
    labels = np.argmin(distances_to_medoids, axis=1)
    return np.asarray(labels, dtype=int)


def compute_total_cost(distance_matrix: np.ndarray, medoids: np.ndarray, labels: np.ndarray) -> float:
    cost = 0.0
    for cluster_id, medoid_idx in enumerate(medoids):
        cluster_points = np.where(labels == cluster_id)[0]
        cost += distance_matrix[cluster_points, medoid_idx].sum()
    return float(cost)


def update_medoids(distance_matrix: np.ndarray, labels: np.ndarray, current_medoids: np.ndarray) -> np.ndarray:
    new_medoids = current_medoids.copy()

    for cluster_id in range(len(current_medoids)):
        cluster_points = np.where(labels == cluster_id)[0]

        if len(cluster_points) == 0:
            continue

        cluster_distances = distance_matrix[np.ix_(cluster_points, cluster_points)]
        total_distances = cluster_distances.sum(axis=1)
        best_local_index = int(np.argmin(total_distances))
        new_medoids[cluster_id] = int(cluster_points[best_local_index])

    return np.asarray(new_medoids, dtype=int)


def pam_single_run(
    distance_matrix: np.ndarray,
    k: int,
    random_state: int = 42,
    max_iter: int = 200
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(random_state)
    n = distance_matrix.shape[0]

    medoids = np.asarray(rng.choice(n, size=k, replace=False), dtype=int)

    for _ in range(max_iter):
        labels = assign_points(distance_matrix, medoids)
        new_medoids = update_medoids(distance_matrix, labels, medoids)

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
# 8. Run PAM for k = 2, 3, 4, 5, 6
# ------------------------------------------------------------

k_values = [2, 3, 4, 5, 6]

evaluation_rows = []
medoid_rows = []
all_cluster_means = {}

df_assignments = pd.DataFrame({
    "Respondent_Row": np.arange(1, len(X_aligned) + 1)
})

for k in k_values:
    print(f"\nRunning PAM-Gower for k = {k} ...")

    medoids, labels, cost = pam_best_of_n(
        distance_matrix=D,
        k=k,
        n_init=30,
        base_random_state=42,
        max_iter=200
    )

    sil = silhouette_score(D, labels, metric="precomputed")

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_size_map = {int(lbl) + 1: int(cnt) for lbl, cnt in zip(unique_labels, counts)}

    smallest_cluster = min(cluster_size_map.values())
    largest_cluster = max(cluster_size_map.values())

    evaluation_rows.append({
        "k": k,
        "Average_Silhouette": round(float(sil), 4),
        "Total_Cost": round(float(cost), 6),
        "Smallest_Cluster_Size": smallest_cluster,
        "Largest_Cluster_Size": largest_cluster,
        "Cluster_Sizes": str(cluster_size_map)
    })

    labels_1based = labels + 1
    df_assignments[f"Cluster_k{k}"] = labels_1based

    for cluster_id, medoid_idx in enumerate(medoids, start=1):
        medoid_rows.append({
            "k": k,
            "Cluster": cluster_id,
            "Medoid_Row": int(medoid_idx) + 1
        })

    df_with_labels = X_aligned.copy()
    df_with_labels["Cluster"] = labels_1based

    cluster_means = df_with_labels.groupby("Cluster").mean(numeric_only=True)
    cluster_means.insert(0, "Cluster_Size", df_with_labels["Cluster"].value_counts().sort_index())
    cluster_means = cluster_means.reset_index()

    all_cluster_means[k] = cluster_means

    print(f"k = {k} complete | silhouette = {sil:.4f} | sizes = {cluster_size_map}")


# ------------------------------------------------------------
# 9. Save outputs
# ------------------------------------------------------------

df_evaluation = pd.DataFrame(evaluation_rows)
df_medoids = pd.DataFrame(medoid_rows)

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_evaluation.to_excel(writer, sheet_name="Evaluation_Summary", index=False)
    df_assignments.to_excel(writer, sheet_name="Assignments", index=False)
    df_medoids.to_excel(writer, sheet_name="Medoids", index=False)

    for k, df_means in all_cluster_means.items():
        df_means.to_excel(writer, sheet_name=f"Cluster_Means_k{k}", index=False)

print(f"\nResults saved to: {output_file.resolve()}")