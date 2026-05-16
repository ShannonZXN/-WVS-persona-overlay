# ============================================================
# Clean selected WVS variables by converting error/missing codes to NaN
#
# INPUT FILES:
#   1) Selected questions_variables.xlsx
#   2) Question and ratings.xlsx
#
# OUTPUT:
#   - Cleaned Excel file with missing/error codes replaced by NaN
#   - Summary table showing which codes were replaced for each question
# ============================================================

from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# 1. File paths
# ============================================================

# Change these if your files are in a different folder
selected_file = Path("Selected_questions_variables_20260331_1stRePro.xlsx")
ratings_file = Path("Question and ratings_20260331.xlsx")

# Output files
cleaned_output_file = Path("Selected_questions_variables_1stRePro_cleaned.xlsx")
summary_output_file = Path("Missing_code_replacement_summary_1stRePro.xlsx")


# ============================================================
# 2. Read the two Excel files
# ============================================================

# Selected survey data: one column per question number
df_selected = pd.read_excel(selected_file, sheet_name="Selected Variables")

# Coding / metadata file: one row per rating code
df_ratings = pd.read_excel(ratings_file, sheet_name=0)

# Standardise column names just in case there are spaces
df_selected.columns = df_selected.columns.astype(str).str.strip()
df_ratings.columns = df_ratings.columns.astype(str).str.strip()


# ============================================================
# 3. Basic checks
# ============================================================

required_rating_columns = [
    "Question_Number",
    "Question_Text",
    "Rating_Code",
    "Rating_Label",
    "Is_Error_Code"
]

missing_required_cols = [col for col in required_rating_columns if col not in df_ratings.columns]
if missing_required_cols:
    raise ValueError(
        f"These required columns are missing from the ratings file: {missing_required_cols}"
    )

print("Selected data shape:", df_selected.shape)
print("Ratings data shape:", df_ratings.shape)
print("Selected data columns:")
print(df_selected.columns.tolist())


# ============================================================
# 4. Build a dictionary:
#    question number -> list of codes that should become NaN
# ============================================================

# Ensure Question_Number is string and stripped
df_ratings["Question_Number"] = df_ratings["Question_Number"].astype(str).str.strip()

# Convert Is_Error_Code safely to boolean
# Handles cases where the column may contain True/False, 1/0, or text
df_ratings["Is_Error_Code"] = df_ratings["Is_Error_Code"].astype(str).str.strip().str.lower().map(
    {"true": True, "false": False, "1": True, "0": False}
).fillna(df_ratings["Is_Error_Code"])

# Keep only rows marked as error / missing / non-substantive
df_error_codes = df_ratings[df_ratings["Is_Error_Code"] == True].copy()

# Make sure Rating_Code is numeric where possible
df_error_codes["Rating_Code_Numeric"] = pd.to_numeric(df_error_codes["Rating_Code"], errors="coerce")

# Build a mapping:
#   error_code_map["Q1"] = [-5, -4, -2, -1]
error_code_map = (
    df_error_codes
    .groupby("Question_Number")["Rating_Code_Numeric"]
    .apply(lambda s: sorted([x for x in s.dropna().unique()]))
    .to_dict()
)


# ============================================================
# 5. Replace error/missing codes with NaN in the selected dataset
# ============================================================

# Keep a record of what we replaced
replacement_summary = []

# Make a copy so the original dataframe stays untouched
df_cleaned = df_selected.copy()

for question_number in df_cleaned.columns:
    if question_number not in error_code_map:
        # No error-code definition found for this variable
        replacement_summary.append({
            "Question_Number": question_number,
            "Question_Text": None,
            "Error_Codes_Found": None,
            "N_Replaced": 0,
            "Note": "No error-code mapping found in ratings file"
        })
        continue

    # Error/missing codes for this question
    codes_to_nan = error_code_map[question_number]

    # Get question text for summary
    text_rows = df_ratings.loc[df_ratings["Question_Number"] == question_number, "Question_Text"]
    question_text = text_rows.iloc[0] if len(text_rows) > 0 else None

    # Count how many values will be replaced
    mask_replace = df_cleaned[question_number].isin(codes_to_nan)
    n_replaced = int(mask_replace.sum())

    # Replace those values with NaN
    df_cleaned.loc[mask_replace, question_number] = np.nan

    replacement_summary.append({
        "Question_Number": question_number,
        "Question_Text": question_text,
        "Error_Codes_Found": ", ".join(str(int(x)) if float(x).is_integer() else str(x) for x in codes_to_nan),
        "N_Replaced": n_replaced,
        "Note": "Replaced with NaN"
    })


# ============================================================
# 6. Create a summary dataframe
# ============================================================

df_summary = pd.DataFrame(replacement_summary)

# Also calculate missing count after cleaning
missing_after_cleaning = []
for question_number in df_cleaned.columns:
    missing_after_cleaning.append({
        "Question_Number": question_number,
        "Missing_Count_After_Cleaning": int(df_cleaned[question_number].isna().sum()),
        "Missing_Percent_After_Cleaning": round(df_cleaned[question_number].isna().mean() * 100, 2)
    })

df_missing_after = pd.DataFrame(missing_after_cleaning)

# Merge summary + missingness
df_summary = df_summary.merge(df_missing_after, on="Question_Number", how="left")


# ============================================================
# 7. Print a quick console summary
# ============================================================

print("\n===== Replacement summary =====")
print(df_summary[["Question_Number", "Error_Codes_Found", "N_Replaced", "Missing_Count_After_Cleaning", "Missing_Percent_After_Cleaning"]])

print("\nTotal missing values after cleaning:", int(df_cleaned.isna().sum().sum()))


# ============================================================
# 8. Save outputs
# ============================================================

# Save cleaned dataset
with pd.ExcelWriter(cleaned_output_file, engine="openpyxl") as writer:
    df_cleaned.to_excel(writer, sheet_name="Selected Variables Cleaned", index=False)

# Save summary
with pd.ExcelWriter(summary_output_file, engine="openpyxl") as writer:
    df_summary.to_excel(writer, sheet_name="Replacement Summary", index=False)

print(f"\nCleaned data saved to: {cleaned_output_file.resolve()}")
print(f"Summary saved to: {summary_output_file.resolve()}")