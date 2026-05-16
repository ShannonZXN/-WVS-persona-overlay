import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# File paths
# ----------------------------
data_file = Path("/mnt/data/Selected_questions_variables_Test1_final.xlsx")
ratings_file = Path("/mnt/data/Question and ratings_20260331.xlsx")
table1_file = Path("/mnt/data/Table1.xlsx")
output_file = Path("/mnt/data/Test1_k4_tables_with_urban_rural.xlsx")

# ----------------------------
# Read files
# ----------------------------
df = pd.read_excel(data_file, sheet_name="Imputed Data")
ratings = pd.read_excel(ratings_file, sheet_name=0)
table1 = pd.read_excel(table1_file, sheet_name=0)

df.columns = df.columns.astype(str).str.strip()
ratings.columns = ratings.columns.astype(str).str.strip()
table1.columns = table1.columns.astype(str).str.strip()

cluster_col = "Cluster_k4"

# ----------------------------
# Variable groups
# ----------------------------
cluster_vars = table1.loc[table1["Role"].astype(str).str.lower() == "cluster", "Question Number"].tolist()
profile_vars = table1.loc[table1["Role"].astype(str).str.lower() == "profile", "Question Number"].tolist()
aux_vars = ["Q260", "Q262", "Q273", "Q275", "H_URBRURAL"]

# Domain mapping for the 20 clustering variables
domain_map = dict(zip(table1["Question Number"], table1["Domain"]))
question_text_map = (
    ratings.groupby("Question_Number")["Question_Text"]
    .first()
    .to_dict()
)

# Rating label maps
valid_ratings = ratings[ratings["Is_Error_Code"] == False].copy()
rating_label_map = {}
for q, sub in valid_ratings.groupby("Question_Number"):
    mapping = {}
    for _, row in sub.iterrows():
        try:
            code = int(row["Rating_Code"])
            mapping[code] = str(row["Rating_Label"])
        except Exception:
            pass
    rating_label_map[q] = mapping

# Valid code ranges and directions for scaling/alignment
valid_code_map = {}
direction_map = {}
for q, sub in ratings.groupby("Question_Number"):
    sub_valid = sub[sub["Is_Error_Code"] == False].copy()
    vals = pd.to_numeric(sub_valid["Rating_Code"], errors="coerce").dropna().astype(float).tolist()
    if vals:
        valid_code_map[q] = sorted(vals)
    dirs = (
        sub["Scale_Direction"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    dirs = [x for x in dirs if x not in ["nan", "none", ""]]
    direction_map[q] = dirs[0] if dirs else None

# Add respondent id if missing
df = df.copy()
df.insert(0, "Respondent_Row", np.arange(1, len(df) + 1))

# ----------------------------
# 1) Respondent-level cluster table
# ----------------------------
respondent_cols = ["Respondent_Row", cluster_col] + cluster_vars + profile_vars + aux_vars
respondent_cols = [c for c in respondent_cols if c in df.columns]
df_respondent = df[respondent_cols].copy()
df_respondent = df_respondent.rename(columns={cluster_col: "Cluster"})

# ----------------------------
# 2) Cluster summary table
#    Use aligned 0–1 domain means + cluster sizes
# ----------------------------
df_aligned = df[cluster_vars].copy()

for q in cluster_vars:
    df_aligned[q] = pd.to_numeric(df_aligned[q], errors="coerce")
    valid_codes = valid_code_map.get(q, None)
    if not valid_codes:
        continue
    min_code = min(valid_codes)
    max_code = max(valid_codes)
    if direction_map.get(q) == "descending":
        df_aligned[q] = max_code + min_code - df_aligned[q]
    rng = max_code - min_code
    if rng == 0:
        df_aligned[q] = 0.0
    else:
        df_aligned[q] = (df_aligned[q] - min_code) / rng

# Domain-level aligned scores
domain_vars = {}
for q in cluster_vars:
    domain_vars.setdefault(domain_map[q], []).append(q)

df_domains = pd.DataFrame(index=df.index)
for domain, vars_in_domain in domain_vars.items():
    df_domains[domain] = df_aligned[vars_in_domain].mean(axis=1)

df_domains["Cluster"] = df[cluster_col].values

cluster_counts = df[cluster_col].value_counts().sort_index()
cluster_perc = (cluster_counts / len(df) * 100).round(2)

cluster_summary = (
    df_domains.groupby("Cluster")
    .mean()
    .reset_index()
)

cluster_summary.insert(1, "Cluster_Size", cluster_counts.astype(int).tolist())
cluster_summary.insert(2, "Percent_of_Sample", cluster_perc.tolist())

# Round domain means for readability
for c in cluster_summary.columns[3:]:
    cluster_summary[c] = cluster_summary[c].round(3)

# ----------------------------
# 3) Cluster profile table
#    Numeric summaries + key categorical percentages (decoded)
# ----------------------------
profile_rows = []

numeric_profile_vars = ["Q262", "Q46", "Q57", "Q286", "Q288", "Q288R"]
categorical_profile_vars = ["Q260", "Q273", "Q275", "Q289", "Q94R", "Q101R", "Q103R", "H_URBRURAL"]

# Numeric summaries
for var in numeric_profile_vars:
    if var not in df.columns:
        continue
    temp = df.groupby(cluster_col)[var].agg(["mean", "std"]).reset_index()
    for _, row in temp.iterrows():
        profile_rows.append({
            "Cluster": int(row[cluster_col]),
            "Variable": var,
            "Question_Text": question_text_map.get(var, var),
            "Summary_Type": "Numeric",
            "Category_or_Level": "Mean (SD)",
            "Value": f"{row['mean']:.2f} ({row['std']:.2f})"
        })

# Categorical summaries (all levels within cluster)
for var in categorical_profile_vars:
    if var not in df.columns:
        continue
    ct = pd.crosstab(df[cluster_col], df[var], normalize="index") * 100
    count_ct = pd.crosstab(df[cluster_col], df[var])

    for cluster_id in ct.index:
        for level in ct.columns:
            label = rating_label_map.get(var, {}).get(int(level), str(level))
            profile_rows.append({
                "Cluster": int(cluster_id),
                "Variable": var,
                "Question_Text": question_text_map.get(var, var),
                "Summary_Type": "Categorical",
                "Category_or_Level": label,
                "Value": f"{count_ct.loc[cluster_id, level]:.0f} ({ct.loc[cluster_id, level]:.2f}%)"
            })

cluster_profile = pd.DataFrame(profile_rows)
cluster_profile = cluster_profile.sort_values(["Cluster", "Variable", "Summary_Type", "Category_or_Level"]).reset_index(drop=True)

# ----------------------------
# 4) Persona summary table
# ----------------------------
persona_rows = [
    {
        "Cluster": 1,
        "Persona_Label": "Moderately Connected Mid-Range Adults",
        "Core_Interpretation": "A relatively moderate cluster across the five persona dimensions, without highly polarised attitudes.",
        "Profiling_Tendencies": "Slightly younger than Clusters 3 and 4, most female-skewed, moderate community participation, predominantly urban.",
        "Aged_Care_Interpretation": "Likely to have mixed support preferences and may engage with support pragmatically rather than from a sharply defined orientation."
    },
    {
        "Cluster": 2,
        "Persona_Label": "Stable but Less Community-Engaged Adults",
        "Core_Interpretation": "The largest cluster, combining relative socioeconomic stability with comparatively lower organisational and mutual-aid participation.",
        "Profiling_Tendencies": "Younger-old profile, relatively higher income and education, lowest church and self-help membership, most urban-oriented.",
        "Aged_Care_Interpretation": "May approach care and support in a more individualised way, with fewer community ties to mediate informal support pathways."
    },
    {
        "Cluster": 3,
        "Persona_Label": "Older Community-Oriented but Less Satisfied Adults",
        "Core_Interpretation": "An older cluster with more visible community participation but lower life and financial satisfaction.",
        "Profiling_Tendencies": "Older and more male-skewed, highest church and self-help membership, comparatively lower wellbeing, mostly urban but with some rural presence.",
        "Aged_Care_Interpretation": "May remain socially connected while still experiencing dissatisfaction or insecurity, suggesting need for support beyond simple community presence."
    },
    {
        "Cluster": 4,
        "Persona_Label": "Secure and Engaged Older Adults",
        "Core_Interpretation": "The most positive overall profile, combining favourable wellbeing with relatively high community engagement.",
        "Profiling_Tendencies": "Oldest cluster on average, highest life and financial satisfaction, strongest charitable participation, comparatively highest rural share.",
        "Aged_Care_Interpretation": "Likely to navigate later-life support from a more secure position, with greater capacity to draw on both formal and informal networks."
    }
]
persona_summary = pd.DataFrame(persona_rows)

# ----------------------------
# Save workbook
# ----------------------------
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    df_respondent.to_excel(writer, sheet_name="1_Respondent_Level_Cluster", index=False)
    cluster_summary.to_excel(writer, sheet_name="2_Cluster_Summary", index=False)
    cluster_profile.to_excel(writer, sheet_name="3_Cluster_Profile", index=False)
    persona_summary.to_excel(writer, sheet_name="4_Persona_Summary", index=False)

print(f"Saved workbook to: {output_file}")
print(cluster_summary)
