import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

with open("spam_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Spam Email Classifier")

st.sidebar.header("Enter Email Features")

word_freq_free = st.sidebar.number_input("Word Frequency of 'free'", min_value=0.0, max_value=100.0, value=0.0)
word_freq_money = st.sidebar.number_input("Word Frequency of 'money'", min_value=0.0, max_value=100.0, value=0.0)
word_freq_offer = st.sidebar.number_input("Word Frequency of 'offer'", min_value=0.0, max_value=100.0, value=0.0)
email_length = st.sidebar.number_input("Email Length", min_value=1, max_value=10000, value=100)

input_data = pd.DataFrame([[word_freq_free, word_freq_money, word_freq_offer, email_length]],
                          columns=['word_freq_free', 'word_freq_money', 'word_freq_offer', 'email_length'])

if st.button("Predict"):
    pred_class = model.predict(input_data)[0]
    pred_prob = model.predict_proba(input_data)[0]

    st.write(f"**Predicted Class:** {'Spam' if pred_class==1 else 'Not Spam'}")
    st.write(f"**Predicted Probabilities:** Not Spam: {pred_prob[0]:.2f}, Spam: {pred_prob[1]:.2f}")

if st.checkbox("Show ROC Curve"):
    
    data = pd.read_csv("synthetic_spam_dataset_extended.csv")
    X = data[['word_freq_free', 'word_freq_money', 'word_freq_offer', 'email_length']]
    y = data['spam']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    y_prob = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

