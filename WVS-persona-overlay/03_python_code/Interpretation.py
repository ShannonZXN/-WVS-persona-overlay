# ============================================================
# Profile final clustering solution: Test 1, k = 4
#
# PURPOSE:
#   1. Read the full imputed dataset
#   2. Read Test 1 cluster assignments
#   3. Merge final cluster labels (k = 4)
#   4. Generate profiling tables by cluster
#   5. Save to Excel
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

# Full imputed dataset: clustering + profiling + auxiliary vars
imputed_file = Path("Selected_questions_variables_20260331_imputed.xlsx")

# Test 1 clustering results file
cluster_results_file = Path("PAM_clustering_results_20260403.xlsx")

# Output file
output_file = Path("Test1_k4_cluster_profiling.xlsx")


# ------------------------------------------------------------
# 2. Variable lists
# ------------------------------------------------------------

# 20 clustering variables
clustering_vars = [
    "Q48", "Q108", "Q8", "Q150",
    "Q1", "Q38", "Q58", "Q27",
    "Q71", "Q74", "Q70", "Q81",
    "Q2", "Q59", "Q60", "Q131",
    "Q47", "Q49", "Q53", "Q50"
]

# Profiling variables
profiling_vars = [
    "Q46", "Q57", "Q94R", "Q101R", "Q103R",
    "Q286", "Q288", "Q288R", "Q289"
]

# Auxiliary variables
auxiliary_vars = ["Q260", "Q262", "Q273", "Q275"]

# Numeric / ordinal-style profiling vars
numeric_profile_vars = ["Q46", "Q57", "Q286", "Q288", "Q288R", "Q262"]

# Categorical / binary profiling vars
categorical_profile_vars = ["Q94R", "Q101R", "Q103R", "Q289", "Q260", "Q273", "Q275"]


# ------------------------------------------------------------
# 3. Read files
# ------------------------------------------------------------

df_imputed = pd.read_excel(imputed_file, sheet_name=0)
df_assignments = pd.read_excel(cluster_results_file, sheet_name="Assignments")

df_imputed.columns = df_imputed.columns.astype(str).str.strip()
df_assignments.columns = df_assignments.columns.astype(str).str.strip()

print("Imputed dataset shape:", df_imputed.shape)
print("Assignments shape:", df_assignments.shape)

# Add respondent row id to full imputed data
df_imputed = df_imputed.copy()
df_imputed["Respondent_Row"] = np.arange(1, len(df_imputed) + 1)

# Keep only needed assignment column
if "Cluster_k4" not in df_assignments.columns:
    raise ValueError("Column 'Cluster_k4' not found in assignments file.")

df_assignments_k4 = df_assignments[["Respondent_Row", "Cluster_k4"]].copy()
df_assignments_k4 = df_assignments_k4.rename(columns={"Cluster_k4": "Cluster"})

# Merge
df_final = df_imputed.merge(df_assignments_k4, on="Respondent_Row", how="left")

if df_final["Cluster"].isna().any():
    raise ValueError("Some rows did not receive a cluster label after merging.")

print("Merged dataset shape:", df_final.shape)


# ------------------------------------------------------------
# 4. Cluster overview
# ------------------------------------------------------------

cluster_counts = df_final["Cluster"].value_counts().sort_index()
cluster_percents = (cluster_counts / len(df_final) * 100).round(2)

df_cluster_overview = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Cluster_Size": cluster_counts.values,
    "Percent_of_Sample": cluster_percents.values
})

print("\nCluster overview:")
print(df_cluster_overview)


# ------------------------------------------------------------
# 5. Cluster means for the 20 clustering variables
# ------------------------------------------------------------

for col in clustering_vars:
    df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

df_cluster_means = df_final.groupby("Cluster")[clustering_vars].mean().reset_index()

# Insert cluster size
cluster_sizes_list = (
    df_final["Cluster"]
    .value_counts()
    .sort_index()
    .astype(int)
    .tolist()
)

df_cluster_means.insert(
    1,
    "Cluster_Size",
    cluster_sizes_list
)


# ------------------------------------------------------------
# 6. Numeric profiling summary
# ------------------------------------------------------------

for col in numeric_profile_vars:
    if col in df_final.columns:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

numeric_summary_rows = []

for var in numeric_profile_vars:
    if var not in df_final.columns:
        continue

    grouped = df_final.groupby("Cluster")[var]
    means = grouped.mean()
    sds = grouped.std()

    for cluster_id in sorted(df_final["Cluster"].unique()):
        numeric_summary_rows.append({
            "Variable": var,
            "Cluster": cluster_id,
            "Mean": round(float(means.loc[cluster_id]), 4) if pd.notna(means.loc[cluster_id]) else np.nan,
            "SD": round(float(sds.loc[cluster_id]), 4) if pd.notna(sds.loc[cluster_id]) else np.nan
        })

df_numeric_profile = pd.DataFrame(numeric_summary_rows)


# ------------------------------------------------------------
# 7. Categorical profiling summary
# ------------------------------------------------------------

categorical_summary_rows = []

for var in categorical_profile_vars:
    if var not in df_final.columns:
        continue

    # Treat as categorical string for safer grouping
    series = df_final[var].astype("Int64").astype(str)

    temp = pd.DataFrame({
        "Cluster": df_final["Cluster"],
        "Value": series
    })

    counts = temp.groupby(["Cluster", "Value"]).size().reset_index(name="Count")

    # Add within-cluster percentage
    counts["Cluster_Total"] = counts["Cluster"].map(df_final["Cluster"].value_counts().to_dict())
    counts["Percent_within_Cluster"] = (counts["Count"] / counts["Cluster_Total"] * 100).round(2)
    counts.insert(0, "Variable", var)

    categorical_summary_rows.append(counts)

df_categorical_profile = pd.concat(categorical_summary_rows, ignore_index=True)


# ------------------------------------------------------------
# 8. Optional wide-format categorical tables
# ------------------------------------------------------------

wide_categorical_tables = {}

for var in categorical_profile_vars:
    if var not in df_final.columns:
        continue

    temp = pd.DataFrame({
        "Cluster": df_final["Cluster"],
        "Value": df_final[var].astype("Int64").astype(str)
    })

    counts = pd.crosstab(temp["Value"], temp["Cluster"])
    percents = pd.crosstab(temp["Value"], temp["Cluster"], normalize="columns") * 100

    # Combine count + percent into one readable table
    combined = counts.copy().astype(str)

    for col in combined.columns:
        combined[col] = counts[col].astype(str) + " (" + percents[col].round(2).astype(str) + "%)"

    combined.index.name = var
    wide_categorical_tables[var] = combined.reset_index()


# ------------------------------------------------------------
# 9. Save outputs
# ------------------------------------------------------------

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_cluster_overview.to_excel(writer, sheet_name="Cluster_Overview", index=False)
    df_cluster_means.to_excel(writer, sheet_name="Cluster_Means_20Vars", index=False)
    df_numeric_profile.to_excel(writer, sheet_name="Numeric_Profile", index=False)
    df_categorical_profile.to_excel(writer, sheet_name="Categorical_Profile_Long", index=False)

    for var, table in wide_categorical_tables.items():
        sheet_name = f"{var}_Profile"
        if len(sheet_name) > 31:
            sheet_name = sheet_name[:31]
        table.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nCluster profiling results saved to: {output_file.resolve()}")