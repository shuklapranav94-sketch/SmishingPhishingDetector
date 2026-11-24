import zipfile
import os
import pandas as pd

DATA_DIR = "data"

zip_files = [
    "archive.zip",
    "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION.zip",
    "sms+spam+collection.zip",
]

for zip_name in zip_files:
    zip_path = os.path.join(DATA_DIR, zip_name)
    extract_folder = os.path.join(DATA_DIR, zip_name.replace(".zip", ""))

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_folder)

    print(f"Extracted: {zip_name} -> {extract_folder}")

# The Mendeley dataset has a zip nested INSIDE the zip we just extracted.
# Build the path to it step by step, then extract that too.
mendeley_folder = os.path.join(
    DATA_DIR,
    "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION",
    "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION",
)
nested_zip_path = os.path.join(mendeley_folder, "Dataset_5971.zip")
nested_extract_folder = os.path.join(mendeley_folder, "Dataset_5971")

with zipfile.ZipFile(nested_zip_path, "r") as z:
    z.extractall(nested_extract_folder)

print(f"Extracted nested zip -> {nested_extract_folder}")





# ---- Load the UCI SMS dataset ----
uci_path = os.path.join(DATA_DIR, "sms+spam+collection", "SMSSpamCollection")
uci = pd.read_csv(uci_path, sep="\t", header=None, names=["label", "text"])

print("\n--- UCI SMS dataset ---")
print("Shape:", uci.shape)
print("\nLabel counts:")
print(uci["label"].value_counts())
print("\nMissing values:")
print(uci.isnull().sum())
print("\nDuplicate rows:", uci.duplicated().sum())





# ---- Load the Mendeley SMS dataset ----
mendeley_csv_path = os.path.join(
    DATA_DIR,
    "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION",
    "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION",
    "Dataset_5971",
    "Dataset_5971.csv",
)
mendeley = pd.read_csv(mendeley_csv_path)
mendeley.columns = [c.strip().lower() for c in mendeley.columns]  # normalize column names

print("\n--- Mendeley SMS dataset ---")
print("Shape:", mendeley.shape)
print("\nColumns:", list(mendeley.columns))
print("\nLabel counts:")
print(mendeley["label"].str.lower().str.strip().value_counts())
print("\nMissing values:")
print(mendeley.isnull().sum())
print("\nDuplicate rows:", mendeley.duplicated().sum())





# ---- Load the email dataset ----
email_path = os.path.join(DATA_DIR, "archive", "phishing_email.csv")
email = pd.read_csv(email_path)

print("\n--- Email dataset ---")
print("Shape:", email.shape)
print("\nColumns:", list(email.columns))
print("\nLabel counts:")
print(email["label"].value_counts())
print("\nMissing values:")
print(email.isnull().sum())
print("\nDuplicate texts:", email.duplicated(subset=["text_combined"]).sum())