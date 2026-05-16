# ============================================================
# Rescale the 20 clustering variables to 0-1
#
# INPUT FILES:
#   1) Selected_questions_variables_20260331_the20variables.xlsx
#   2) Question and ratings_20260331.xlsx
#
# OUTPUT:
#   1) Selected_questions_variables_20260331_the20variables_scaled_01.xlsx
#   2) Scaling_summary_20variables.xlsx
#
# PURPOSE:
#   - Use valid codes from the ratings file
#   - Reverse descending scales so direction is consistent
#   - Rescale all 20 variables to the range [0, 1]
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

input_data_file = Path("Selected_questions_variables_20260406_the20variables.xlsx")
ratings_file = Path("Question and ratings_20260331.xlsx")

scaled_output_file = Path("Selected_questions_variables_20260406_the20variables_scaled_01.xlsx")
summary_output_file = Path("Scaling_summary_20variables.xlsx")


# ------------------------------------------------------------
# 2. Read Excel files
# ------------------------------------------------------------

df_data = pd.read_excel(input_data_file, sheet_name="Imputed Data")
df_ratings = pd.read_excel(ratings_file, sheet_name=0)

# Standardise column names
df_data.columns = df_data.columns.astype(str).str.strip()
df_ratings.columns = df_ratings.columns.astype(str).str.strip()

print("Input data shape:", df_data.shape)
print("Ratings shape:", df_ratings.shape)
print("Data columns:", df_data.columns.tolist())


# ------------------------------------------------------------
# 3. Basic metadata checks
# ------------------------------------------------------------

required_rating_columns = [
    "Question_Number",
    "Rating_Code",
    "Is_Error_Code",
    "Variable_Type",
    "Scale_Direction"
]

missing_cols = [col for col in required_rating_columns if col not in df_ratings.columns]
if missing_cols:
    raise ValueError(
        f"Missing required columns in ratings file: {missing_cols}"
    )

# Standardise metadata values
df_ratings["Question_Number"] = df_ratings["Question_Number"].astype(str).str.strip()
df_ratings["Variable_Type"] = df_ratings["Variable_Type"].astype(str).str.strip().str.lower()
df_ratings["Scale_Direction"] = df_ratings["Scale_Direction"].astype(str).str.strip().str.lower()

# Convert Is_Error_Code to boolean safely
df_ratings["Is_Error_Code"] = (
    df_ratings["Is_Error_Code"]
    .astype(str).str.strip().str.lower()
    .map({"true": True, "false": False, "1": True, "0": False})
    .fillna(False)  # unrecognised → not an error code
    .astype(bool)
)

# Numeric rating codes only
df_ratings["Rating_Code_Numeric"] = pd.to_numeric(df_ratings["Rating_Code"], errors="coerce")


# ------------------------------------------------------------
# 4. Build valid-code metadata for each question
# ------------------------------------------------------------

questions = df_data.columns.tolist()

valid_code_map = {}
variable_type_map = {}
direction_map = {}

for q in questions:
    sub = df_ratings[df_ratings["Question_Number"] == q].copy()

    # valid codes = non-error codes only
    valid_codes_series = (
        sub.loc[sub["Is_Error_Code"] == False, "Rating_Code_Numeric"]
        .dropna()
        .astype("float64")
    )
    valid_codes_array = valid_codes_series.to_numpy(dtype=float)
    valid_codes = sorted(valid_codes_array.tolist())

    if len(valid_codes) == 0:
        raise ValueError(f"No valid rating codes found in ratings file for {q}")

    valid_code_map[q] = valid_codes

    # Variable type
    type_values = (
        sub["Variable_Type"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )
    variable_type_map[q] = type_values[0] if len(type_values) > 0 else None

    # Scale direction
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


# ------------------------------------------------------------
# 5. Rescale each variable to 0-1
# ------------------------------------------------------------

df_scaled = df_data.copy()
scaling_summary = []

for q in questions:
    # Convert column to numeric
    df_scaled[q] = pd.to_numeric(df_scaled[q], errors="coerce")

    valid_codes = valid_code_map[q]
    min_code = min(valid_codes)
    max_code = max(valid_codes)
    var_type = variable_type_map[q]
    direction = direction_map[q]

    # Safety check: ensure all observed values are inside valid code set
    observed_values_series = df_scaled[q].dropna().astype("float64")
    observed_values_array = observed_values_series.to_numpy(dtype=float)
    observed_values = sorted(pd.unique(observed_values_array).tolist())
    invalid_values = [x for x in observed_values if x not in valid_codes]

    if invalid_values:
        raise ValueError(
            f"{q} contains values not in valid code list.\n"
            f"Observed invalid values: {invalid_values}\n"
            f"Valid codes: {valid_codes}"
        )

    # If only one valid code exists, cannot scale normally
    if max_code == min_code:
        df_scaled[q] = 0.0
        note = "Single valid code only; set to 0.0"
    else:
        # Standard min-max scaling
        df_scaled[q] = (df_scaled[q] - min_code) / (max_code - min_code)

        # Reverse if descending so higher scaled values always mean "more"
        if direction == "descending":
            df_scaled[q] = 1.0 - df_scaled[q]
            note = "Scaled 0-1 and reverse-coded"
        else:
            note = "Scaled 0-1"

    scaling_summary.append({
        "Question_Number": q,
        "Variable_Type": var_type,
        "Scale_Direction": direction,
        "Valid_Codes": str(valid_codes),
        "Min_Code": min_code,
        "Max_Code": max_code,
        "Action": note,
        "Scaled_Min": float(df_scaled[q].min()),
        "Scaled_Max": float(df_scaled[q].max())
    })


# ------------------------------------------------------------
# 6. Optional quick checks
# ------------------------------------------------------------

print("\n===== Scaling checks =====")
for q in questions:
    print(f"{q}: min={df_scaled[q].min():.3f}, max={df_scaled[q].max():.3f}")

# Check if any values fall outside [0,1]
outside_range = []
for q in questions:
    if (df_scaled[q] < 0).any() or (df_scaled[q] > 1).any():
        outside_range.append(q)

if outside_range:
    print("\nWarning: these variables have values outside [0,1]:", outside_range)
else:
    print("\nAll variables successfully scaled to [0,1].")


# ------------------------------------------------------------
# 7. Save outputs
# ------------------------------------------------------------

df_summary = pd.DataFrame(scaling_summary)

with pd.ExcelWriter(scaled_output_file, engine="openpyxl") as writer:
    df_scaled.to_excel(writer, sheet_name="Scaled_0_1", index=False)

with pd.ExcelWriter(summary_output_file, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Scaling Summary", index=False)

print(f"\nScaled dataset saved to: {scaled_output_file.resolve()}")
print(f"Scaling summary saved to: {summary_output_file.resolve()}")