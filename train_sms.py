import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---- Load the cleaned dataset ----
df = pd.read_csv("data/sms_combined_clean.csv")
df["label_num"] = (df["binary_label"] == "spam").astype(int)  # 1 = spam, 0 = ham

X = df["text"]
y = df["label_num"]

# ---- Split into train and test ----
# stratify=y keeps the same spam/ham ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", len(X_train), "| Train spam %:", round(y_train.mean() * 100, 1))
print("Test size:", len(X_test), "| Test spam %:", round(y_test.mean() * 100, 1))

# ---- Convert text into numbers (TF-IDF) ----
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),  # single words AND two-word phrases
    min_df=2,            # ignore words that appear in fewer than 2 messages
    max_df=0.95,         # ignore words that appear in more than 95% of messages
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nVocabulary size:", len(vectorizer.vocabulary_))

# ---- Train the model ----
# class_weight="balanced" tells the model to pay more attention to spam,
# since spam is the minority class (17.5% of the data)
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_vec, y_train)

# ---- Evaluate on the test set (data the model never saw during training) ----
predictions = model.predict(X_test_vec)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\n", classification_report(y_test, predictions, target_names=["ham", "spam"], digits=3))

# ---- Save the trained model and vectorizer ----
joblib.dump(vectorizer, "models/sms_tfidf_vectorizer.joblib")
joblib.dump(model, "models/sms_logreg_model.joblib")
print("Saved model and vectorizer to models/")