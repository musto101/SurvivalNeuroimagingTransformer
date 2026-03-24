import pandas as pd

dat = pd.read_csv('data/raw_data.csv')

dat.head()
dat.columns

dat_trimmed = dat.drop(columns=["vis_01_dt", "ace3_34_ace3tot",
                                 'acer_36_mmsetot', "acer_37_acertot"])

dat_trimmed.columns

# Example: your input dataframe
# df = pd.read_csv("your_file.csv")

# Replace these with your real column names
id_col = "record_id"

test_mappings = {
    "atttot_combined": ("ace3_35_atttot", "acer_38_atttot"),
    "memtot_combined": ("ace3_36_memtot", "acer_39_memtot"),
    "fluentot_combined": ("ace3_37_fluentot", "acer_40_fluentot"),
    "langtot_combined": ("ace3_38_langtot", "acer_41_langtot"),
    "visuisptot_combined": ("ace3_39_visuosptot", "acer_42_visuosptot")
}

# Start new dataframe with participant ID
new_dat = dat_trimmed[[id_col]].copy()

# Create combined columns
for new_col, (orig_col, rev_col) in test_mappings.items():
    new_dat[new_col] = dat_trimmed[orig_col].combine_first(dat_trimmed[rev_col])

# View result
print(new_dat.head())

new_dat

################### data cleaning for fake data

dat = pd.read_csv("data/fake_data/mgus2_june_paper_placeholder.csv")

# check for missing values
print(dat.isnull().sum())

# missing values as a percentage of total for each column
print(dat.isnull().mean() * 100)

# create new columns indicating missingness for baseline_biomarker columns
for col in dat.columns:
    if col.startswith("baseline_biomarker"):
        dat[f"{col}_missing"] = dat[col].isnull().astype(int)

# view the new columns
print(dat.filter(like="baseline_biomarker").head())

dat.columns

# check the time columns for max values
print(dat["time_to_primary_event_months"].max())
print(dat["time_to_competing_event_or_censor_months"].max())

# save the cleaned data to a new CSV file
dat.to_csv("data/fake_data/mgus2_june_paper_placeholder_cleaned.csv", index=False)