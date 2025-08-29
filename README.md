# 📧 Spam Detection using Naïve Bayes

## 📝 Project Description

This project implements a Spam Detection System using Naïve Bayes Classification.

The goal is to classify text messages (emails or SMS) as spam or not spam based on word frequency features such as "free", "money", "offer".

The model is trained and evaluated on a synthetic dataset to demonstrate the process of building a spam filter.

## 📂 Dataset

Dataset used: synthetic_spam_dataset_extended.csv

Features include:

📌 word_freq_free → frequency of word "free"

📌 word_freq_money → frequency of word "money"

📌 word_freq_offer → frequency of word "offer"

📌 char_freq_$ → frequency of symbol "$"

Target column: label → (1 = Spam, 0 = Not Spam)

## ⚙️ Requirements

Install the following Python libraries:

🐼 pandas

📊 matplotlib

🎨 seaborn

🤖 scikit-learn

💾 pickle (for saving/loading models)

## 🚀 How to Run

📥 Clone the repository or download the files.

📂 Place the dataset (synthetic_spam_dataset_extended.csv) in the same directory.

🖥️ Open and run the Jupyter Notebook:

jupyter notebook spam.ipynb


The notebook performs:

🔍 Data exploration (EDA)

🧹 Preprocessing (check nulls, duplicates)

✂️ Train-test split

🧠 Training with Gaussian Naïve Bayes

📈 Model evaluation (accuracy, confusion matrix, classification report, ROC curve)

💾 Saving the model using pickle

## 📊 Results

✅ The Gaussian Naïve Bayes classifier achieved high accuracy on the dataset.

📑 Evaluation metrics used:

🎯 Accuracy Score

📊 Precision, Recall, F1-score (Classification Report)

🗂️ Confusion Matrix

📉 ROC Curve & AUC Score

<img width="691" height="545" alt="image" src="https://github.com/user-attachments/assets/edca7a5e-031e-4be2-b4a9-511a147d4ee7" />


## ✅ Conclusion

✨ The Naïve Bayes classifier is a simple yet effective method for spam detection.

📩 Using word frequency features, the model successfully distinguishes between spam and non-spam.

📈 Gaussian Naïve Bayes was chosen because the features are continuous word frequencies, not binary or raw counts.

## 🔮 Future Development

🌍 Use real-world datasets (SMS Spam Collection, Enron Email).

🔄 Try other Naïve Bayes variants: MultinomialNB, BernoulliNB.

🧩 Feature engineering: message length, links, suspicious domains.

🤝 Compare with Logistic Regression, Random Forest, Gradient Boosting, Deep Learning (LSTMs, BERT).

🌐 Deploy as a web app using Flask or Streamlit.

## 📁 Files

📓 spam.ipynb → Main Jupyter Notebook

📄 synthetic_spam_dataset_extended.csv → Dataset

💾 spam_model.pkl → Saved model
