import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ---- Load the cleaned dataset ----
df = pd.read_csv("data/email_clean.csv")
df = df.dropna(subset=["text_combined"])  # just in case any rows lost their text

X = df["text_combined"]
y = df["label"]  # 1 = phishing, 0 = legit

# ---- Split into train and test ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", len(X_train), "| Train phishing %:", round(y_train.mean() * 100, 1))
print("Test size:", len(X_test), "| Test phishing %:", round(y_test.mean() * 100, 1))

# ---- Convert text into numbers (TF-IDF) ----
# max_features caps the vocabulary size, since email text is much larger than SMS
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=50000,
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("\nVocabulary size:", len(vectorizer.vocabulary_))

# ---- Train the model ----
model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
model.fit(X_train_vec, y_train)

# ---- Evaluate ----
predictions = model.predict(X_test_vec)

print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\n", classification_report(y_test, predictions, target_names=["legit", "phishing"], digits=3))

# ---- Save ----
joblib.dump(vectorizer, "models/email_tfidf_vectorizer.joblib")
joblib.dump(model, "models/email_logreg_model.joblib")
print("Saved model and vectorizer to models/")