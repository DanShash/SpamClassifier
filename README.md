README.md:

# Email Spam Classifier with NLP

A machine learning project to classify emails as spam or ham using the SpamAssassin Public Corpus. Built with Scikit-learn, NLTK, and custom NLP preprocessing, this project achieves **98.5% accuracy** in detecting spam emails.

## Project Overview

This project implements an NLP-based spam email classifier using logistic regression. It processes 3,000 emails (2,500 ham, 500 spam) from the SpamAssassin Public Corpus, applying custom preprocessing to handle HTML, multipart formats, and imbalanced data. The model is trained via a Scikit-learn pipeline and achieves high performance: **98.5% accuracy**, **96.88% precision**, and **97.89% recall**.

### Features
- **Dataset**: 3,000 emails from SpamAssassin (2,500 ham, 500 spam).
- **Preprocessing**: NLTK Porter Stemming, URL/number replacement, word vectorization, and HTML-to-text conversion.
- **Model**: Logistic regression with a custom Scikit-learn pipeline.
- **Performance**: 98.5% cross-validation accuracy, 96.88% precision, 97.89% recall on the test set.
- **Deployment**: Optional Flask API for real-time spam classification (see `app.py`).

## Setup Instructions

### Prerequisites
- Python 3.7+
- Git
- Internet connection (to download the dataset)
- scikit-learn>=1.0.1
- nltk
- urlextract
- pandas
- flask (for API)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/[your-repo-name].git
   cd [your-repo-name]

Install dependencies:
bash

pip install -r requirements.txt

Download and extract the SpamAssassin dataset:
Run the notebook (spam_classifier.ipynb) to automatically fetch 20030228_easy_ham.tar.bz2 and 20030228_spam.tar.bz2 to datasets/spam/.

Alternatively, manually download from SpamAssassin Public Corpus.

Running the Notebook
Launch Jupyter Notebook:
bash

jupyter notebook

Open spam_classifier.ipynb and run all cells to:
Fetch and preprocess the dataset.

Train the logistic regression model.

Evaluate performance (accuracy, precision, recall).

The trained model is saved as spam_classifier.pkl.

Running the Flask API (Optional)
Ensure the trained model (spam_classifier.pkl) is in the repo root.

Run the Flask app:
bash

python app.py

Access the API at http://localhost:5000/predict via a POST request (see Flask API Usage below).

Dataset
Source: SpamAssassin Public Corpus

Files:
20030228_easy_ham.tar.bz2: 2,500 non-spam emails.

20030228_spam.tar.bz2: 500 spam emails.

Size: 3,000 emails (16.7% spam ratio, imbalanced).

Structure: Emails include plain text, HTML, and multipart formats, requiring robust preprocessing.

Performance
Cross-Validation Accuracy: 98.5% (3-fold CV on training set).

Test Set Metrics:
Precision: 96.88%

Recall: 97.89%

Visualizations: See images/confusion_matrix.png for the test set confusion matrix.

Flask API Usage
The Flask API (app.py) allows real-time spam classification via HTTP requests.
Endpoint
URL: POST /predict

Payload: JSON with an email field containing the email text (plain text or HTML).
json

{
  "email": "Subject: Win a free prize!\nClick here: http://scam.com to claim your reward!"
}

Response: JSON with prediction and probability.
json

{
  "prediction": "spam",
  "probability": 0.95
}

Example Request (using curl)
bash

curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"email": "Subject: Win a free prize!\nClick here: http://scam.com to claim your reward!"}'

Files
spam_classifier.ipynb: Jupyter Notebook with the full project code.

app.py: Flask API script for deployment (optional).

spam_classifier.pkl: Trained logistic regression model.

requirements.txt: Python dependencies.

images/confusion_matrix.png: Confusion matrix visualization.

datasets/spam/: Directory for SpamAssassin dataset (created during execution).

### Future Improvements
Incorporate TF-IDF or BERT for enhanced NLP performance.

Address dataset imbalance using SMOTE or class weighting.

Deploy the Flask API on a cloud platform (e.g., Heroku, AWS).

Acknowledgments
Dataset: Apache SpamAssassin Public Corpus

Libraries: Scikit-learn, NLTK, urlextract
