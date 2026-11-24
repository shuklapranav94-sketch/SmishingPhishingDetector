import os
import pandas as pd

DATA_DIR = "data"

email_path = os.path.join(DATA_DIR, "archive", "phishing_email.csv")
email = pd.read_csv(email_path)

print("Before removing duplicates:", len(email))

email = email.drop_duplicates(subset=["text_combined"], keep="first").reset_index(drop=True)

print("After removing duplicates:", len(email))
print(email["label"].value_counts())

email.to_csv("data/email_clean.csv", index=False)
print("\nSaved to data/email_clean.csv")