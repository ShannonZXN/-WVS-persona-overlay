# ============================================================
# MissForest imputation for selected WVS variables
#
# PURPOSE:
#   1. Read cleaned selected-variable dataset from Excel
#   2. Read variable type metadata from the ratings Excel file
#   3. Use MissForest to impute NaN values
#   4. Save the imputed dataset to a new Excel file
#
# FILES:
#   - Input cleaned data:
#       Selected_questions_variables_cleaned.xlsx
#   - Input metadata:
#       Question and ratings_20260331.xlsx
#   - Output imputed data:
#       Selected_questions_variables_imputed.xlsx
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np
from missforest import MissForest

# ------------------------------------------------------------
# 1. File paths
# ------------------------------------------------------------

# Cleaned dataset from the previous step
cleaned_file_path = Path("Selected_questions_variables_1stRePro_cleaned.xlsx")

# Metadata / coding file
ratings_file_path = Path("Question and ratings_20260331.xlsx")

# Output file
imputed_output_file_path = Path("Selected_questions_variables_1stRePro_imputed.xlsx")


# ------------------------------------------------------------
# 2. Read Excel files
# ------------------------------------------------------------

# Read cleaned selected-variable dataset
df_cleaned = pd.read_excel(cleaned_file_path, sheet_name=0)

# Read ratings / metadata file
df_ratings = pd.read_excel(ratings_file_path, sheet_name=0)

# Standardise column names
df_cleaned.columns = df_cleaned.columns.astype(str).str.strip()
df_ratings.columns = df_ratings.columns.astype(str).str.strip()

print("Cleaned dataset shape:", df_cleaned.shape)
print("Ratings dataset shape:", df_ratings.shape)


# ------------------------------------------------------------
# 3. Check required metadata columns
# ------------------------------------------------------------

# Standardise raw column names first
df_ratings.columns = df_ratings.columns.astype(str).str.strip()

print("Ratings file columns:")
print(df_ratings.columns.tolist())

# Required question-number column
if "Question_Number" not in df_ratings.columns:
    raise ValueError(
        "Column 'Question_Number' not found in ratings file. "
        f"Available columns are: {df_ratings.columns.tolist()}"
    )

# Try to automatically detect the variable-type column
possible_type_columns = [
    "Type",
    "type",
    "Question_Type",
    "Variable_Type",
    "Data Type",
    "Data_Type",
    "Scale_Type",
    "Scale Type"
]

type_column_found = None
for col in possible_type_columns:
    if col in df_ratings.columns:
        type_column_found = col
        break

if type_column_found is None:
    raise ValueError(
        "Could not find the variable type column in ratings file. "
        f"Available columns are: {df_ratings.columns.tolist()}"
    )

print(f"Using '{type_column_found}' as the variable type column.")

# Create a standardised Type column for the rest of the script
df_ratings["Question_Number"] = df_ratings["Question_Number"].astype(str).str.strip()
df_ratings["Type"] = df_ratings[type_column_found].astype(str).str.strip().str.lower()


# ------------------------------------------------------------
# 4. Build variable type lists from metadata
# ------------------------------------------------------------

# Keep metadata only for variables present in cleaned data
df_meta_selected = df_ratings[df_ratings["Question_Number"].isin(df_cleaned.columns)].copy()

# Remove duplicate rows if metadata repeats per question
df_meta_selected = df_meta_selected.drop_duplicates(subset=["Question_Number"])

# Define supported type labels
ordinal_labels = {"ordinal"}
binary_labels = {"binary"}
nominal_labels = {"nominal"}

ordinal_variables = df_meta_selected.loc[
    df_meta_selected["Type"].isin(ordinal_labels), "Question_Number"
].tolist()

binary_variables = df_meta_selected.loc[
    df_meta_selected["Type"].isin(binary_labels), "Question_Number"
].tolist()

nominal_variables = df_meta_selected.loc[
    df_meta_selected["Type"].isin(nominal_labels), "Question_Number"
].tolist()

all_typed_variables = set(ordinal_variables + binary_variables + nominal_variables)
all_data_variables = set(df_cleaned.columns)

untyped_variables = sorted(list(all_data_variables - all_typed_variables))
if untyped_variables:
    print("\nWarning: these variables have no recognised type in metadata:")
    print(untyped_variables)
    print("They will be treated as nominal by default.\n")
    nominal_variables.extend(untyped_variables)

# MissForest categorical variables = binary + nominal
categorical_variables = sorted(list(set(binary_variables + nominal_variables)))

print("Ordinal variables:")
print(ordinal_variables)
print("\nBinary variables:")
print(binary_variables)
print("\nNominal variables:")
print(nominal_variables)
print("\nCategorical variables passed to MissForest:")
print(categorical_variables)


# ------------------------------------------------------------
# 5. Prepare dataframe for imputation
# ------------------------------------------------------------

# Work on a copy
df_impute_input = df_cleaned.copy()

# Convert all columns to numeric where possible
# Missing values remain as NaN
for column_name in df_impute_input.columns:
    df_impute_input[column_name] = pd.to_numeric(
        df_impute_input[column_name],
        errors="coerce"
    )

# Optional safety check
print("\nMissing values before imputation:", int(df_impute_input.isna().sum().sum()))
print("Categorical variables used by MissForest:")
print(categorical_variables)
# ------------------------------------------------------------
# 6. Run MissForest imputation
# ------------------------------------------------------------

# In your installed MissForest version, categorical variables
# must be passed into the constructor, not fit_transform().
imputer = MissForest(
    categorical=categorical_variables,
    initial_guess="median",
    max_iter=5,
    early_stopping=True,
    verbose=2
)

# Fit and transform
df_imputed = imputer.fit_transform(df_impute_input)

# Safety: convert NumPy output back to DataFrame if needed
if isinstance(df_imputed, np.ndarray):
    df_imputed = pd.DataFrame(df_imputed, columns=df_impute_input.columns)

# Preserve original column order
df_imputed = df_imputed[df_cleaned.columns.tolist()]


# ------------------------------------------------------------
# 7. Round coded questionnaire variables back to integer values
# ------------------------------------------------------------

# Since the survey data are coded in discrete categories,
# imputed results should be mapped back to integer codes.

for column_name in df_imputed.columns:
    df_imputed[column_name] = pd.to_numeric(df_imputed[column_name], errors="coerce")
    df_imputed[column_name] = df_imputed[column_name].round().astype("Int64")


# ------------------------------------------------------------
# 8. Optional validity check against known valid codes
# ------------------------------------------------------------

# If your ratings file contains a column like "Is_Error_Code",
# this block checks whether imputed values fall outside the valid codes.

validity_check_available = all(
    col in df_ratings.columns for col in ["Question_Number", "Rating_Code", "Is_Error_Code"]
)

if validity_check_available:
    df_valid_codes = df_ratings.copy()
    df_valid_codes["Is_Error_Code"] = (
        df_valid_codes["Is_Error_Code"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(df_valid_codes["Is_Error_Code"])
    )
    df_valid_codes["Rating_Code_Numeric"] = pd.to_numeric(df_valid_codes["Rating_Code"], errors="coerce")

    valid_code_map = (
        df_valid_codes[df_valid_codes["Is_Error_Code"] == False]
        .groupby("Question_Number")["Rating_Code_Numeric"]
        .apply(lambda s: sorted([int(x) for x in s.dropna().unique()]))
        .to_dict()
    )

    invalid_summary = []

    for column_name in df_imputed.columns:
        if column_name not in valid_code_map:
            continue

        valid_codes = valid_code_map[column_name]
        invalid_mask = ~df_imputed[column_name].isin(valid_codes)
        n_invalid = int(invalid_mask.sum())

        if n_invalid > 0:
            invalid_summary.append({
                "Question_Number": column_name,
                "Valid_Codes": str(valid_codes),
                "N_Invalid_After_Rounding": n_invalid
            })

    if invalid_summary:
        df_invalid_summary = pd.DataFrame(invalid_summary)
        print("\nWarning: some imputed values fall outside valid code ranges after rounding.")
        print(df_invalid_summary)
    else:
        print("\nAll imputed values fall within known valid code ranges.")
else:
    print("\nValidity check skipped because Rating_Code / Is_Error_Code metadata were not fully available.")


# ------------------------------------------------------------
# 9. Missingness summary
# ------------------------------------------------------------

missing_before = int(df_cleaned.isna().sum().sum())
missing_after = int(df_imputed.isna().sum().sum())

print("\n===== Missingness summary =====")
print("Total missing values before imputation:", missing_before)
print("Total missing values after imputation:", missing_after)

if missing_after == 0:
    print("All NaN values have been imputed.")
else:
    print("Warning: some missing values remain after imputation.")


# ------------------------------------------------------------
# 10. Save output to Excel
# ------------------------------------------------------------

with pd.ExcelWriter(imputed_output_file_path, engine="openpyxl") as writer:
    df_imputed.to_excel(writer, sheet_name="Imputed Data", index=False)

print(f"\nImputed dataset saved to: {imputed_output_file_path.resolve()}")