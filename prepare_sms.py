import os
import pandas as pd

DATA_DIR = "data"

# ---- Load UCI ----
uci_path = os.path.join(DATA_DIR, "sms+spam+collection", "SMSSpamCollection")
uci = pd.read_csv(uci_path, sep="\t", header=None, names=["label", "text"])
uci["label"] = uci["label"].str.lower().str.strip()

# ---- Load Mendeley ----
mendeley_path = os.path.join(
    DATA_DIR,
    "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION",
    "SMS PHISHING DATASET FOR MACHINE LEARNING AND PATTERN RECOGNITION",
    "Dataset_5971",
    "Dataset_5971.csv",
)
mendeley = pd.read_csv(mendeley_path)
mendeley.columns = [c.strip().lower() for c in mendeley.columns]
mendeley["label"] = mendeley["label"].str.lower().str.strip()
mendeley = mendeley[["label", "text"]]  # keep only the columns we need

# ---- Collapse to binary: ham vs spam (spam + smishing both become "spam") ----
def collapse(label):
    return "ham" if label == "ham" else "spam"

uci["binary_label"] = uci["label"].apply(collapse)
mendeley["binary_label"] = mendeley["label"].apply(collapse)

# ---- Combine both datasets ----
combined = pd.concat(
    [uci[["text", "binary_label"]], mendeley[["text", "binary_label"]]],
    ignore_index=True,
)

print("Before removing duplicates:", len(combined))
print(combined["binary_label"].value_counts())

# ---- Remove duplicates (case-insensitive, whitespace-trimmed) ----
combined["text_norm"] = combined["text"].str.strip().str.lower()
combined = combined.drop_duplicates(subset=["text_norm"], keep="first")
combined = combined.drop(columns=["text_norm"]).reset_index(drop=True)

print("\nAfter removing duplicates:", len(combined))
print(combined["binary_label"].value_counts())

# ---- Save the cleaned, combined dataset ----
os.makedirs("models", exist_ok=True)  # we'll reuse this folder for outputs too
combined.to_csv("data/sms_combined_clean.csv", index=False)
print("\nSaved to data/sms_combined_clean.csv")