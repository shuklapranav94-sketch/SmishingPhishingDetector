# 🛡️ Smishing & Phishing Detector

An AI-powered web application that detects potentially fraudulent SMS messages and emails using Machine Learning and Natural Language Processing (NLP).

The application analyzes message text and predicts whether it is **Safe, Suspicious, or Spam/Phishing**, helping users identify potentially malicious messages.

## 🚀 Features

- 📱 SMS message detection
- 📧 Email phishing and spam detection
- 🤖 Machine Learning-based classification
- 📝 Text analysis using TF-IDF
- ⚡ Fast prediction using trained models
- 📊 Spam probability score
- 🎨 Clean, responsive web interface
- 🔍 Separate scanning modes for SMS and Email

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Backend & ML |
| Flask | Web framework |
| Scikit-learn | Machine Learning |
| TF-IDF | Text feature extraction |
| HTML5 | Frontend structure |
| CSS3 | Frontend styling |
| Joblib | Saving and loading trained models |
| Pandas | Dataset processing |

## 🧠 Machine Learning

The project uses trained Machine Learning models to classify messages.

### Models Used

- Logistic Regression for classification
- TF-IDF Vectorizer for text feature extraction

Separate trained models and vectorizers are used for SMS and Email detection.

### Detection Process

1. User enters an SMS or email message.
2. The application preprocesses the input text.
3. TF-IDF converts text into numerical features.
4. The trained Machine Learning model analyzes the features.
5. The application displays the prediction and spam probability.

## 📂 Project Structure

```text
SmishingPhishingDetector/
│
├── app.py
├── requirements.txt
├── explore_data.py
├── prepare_email.py
├── prepare_sms.py
├── train_email.py
├── train_sms.py
│
├── models/
│   ├── email_logreg_model.joblib
│   ├── email_tfidf_vectorizer.joblib
│   ├── sms_logreg_model.joblib
│   └── sms_tfidf_vectorizer.joblib
│
├── data/
│   ├── archive/
│   ├── sms+spam+collection/
│   └── other datasets
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/shuklapranav94-sketch/SmishingPhishingDetector.git
```

### 2. Navigate to the project directory

```bash
cd SmishingPhishingDetector
```

### 3. Create a virtual environment

```bash
python3 -m venv venv
```

### 4. Activate the virtual environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the application

```bash
python app.py
```

### 7. Open in your browser

```text
http://127.0.0.1:5001
```

> If the application uses a different port in `app.py`, open the port configured there.

## 📊 Dataset

This project uses SMS spam and email phishing datasets for training and testing Machine Learning models.

### 📥 Download Datasets

The large datasets are hosted on Google Drive because of GitHub file size limitations.

- 📧 **Phishing Email Dataset:** [Download phishing_email.csv](https://drive.google.com/file/d/1HfbnBEae5UiqlMMKaMfM5vhAUhztnE-R/view?usp=drive_link)

- 📊 **Cleaned Email Dataset:** [Download email_clean.csv](https://drive.google.com/file/d/13bE5a86Gbg9PFpsTI6HHHqb2cBVDAIPC/view?usp=drive_link)

### Dataset Usage

These datasets were used for email data processing, feature extraction, and Machine Learning model training.

The trained models are already included in the repository, so users can run the application without downloading the datasets.

---

## 🔮 Future Improvements

- Improve model accuracy using advanced NLP techniques.
- Add URL detection and suspicious link analysis.
- Add explainable AI to explain predictions.
- Deploy the application online.
- Add more phishing datasets.
- Improve detection of new and evolving phishing attacks.
- Add real-time threat intelligence integration.

## ⚠️ Disclaimer

This project is developed for educational and research purposes.

Machine Learning predictions are not guaranteed to be 100% accurate. Always verify suspicious messages independently and avoid sharing sensitive information through untrusted links.

## 👨‍💻 Author

**Pranav Shukla**

GitHub: [@shuklapranav94-sketch](https://github.com/shuklapranav94-sketch)

---

⭐ If you find this project useful, consider giving it a star!
