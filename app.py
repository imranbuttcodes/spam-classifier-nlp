import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import joblib

nltk.download('punkt', quiet= True)
nltk.download('stopwords', quiet= True)

stemmer = PorterStemmer()
cached_stopwords = set(stopwords.words('english')) 

def text_preprocessing(text):
    text = text.lower()
    tokens = word_tokenize(text)
    
    cleaned_tokens = []
    for word in tokens:    
        if word.isalnum() and word not in cached_stopwords:
            cleaned_tokens.append(stemmer.stem(word))
        

    return " ".join(cleaned_tokens)



st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📧",
    layout="centered"
)


model = joblib.load("Model.joblib")
vectorizer = joblib.load("Vectorizer.joblib")

st.title("Email Spam Detection System")


st.subheader("Message Input")

message = st.text_area(
    "Enter your message below:",
    height=150,
    placeholder="Type or paste an email/SMS message here..."
)


predict_btn = st.button("Run Classification", use_container_width=True)


if predict_btn:

    if message.strip() == "":
        st.warning("Input field is empty. Please enter a message.")
    else:
        transform_input = text_preprocessing(message)

        transformed_input = vectorizer.transform([transform_input])

        prediction = model.predict(transformed_input)[0]

        probability = model.predict_proba(transformed_input)[0]

        st.divider()
        st.subheader("Result")

        if prediction == 1:
            st.error("Classification: SPAM")
        else:
            st.success("Classification: NOT SPAM")

        st.subheader("Confidence Score")

        spam_conf = probability[1]
        ham_conf = probability[0]

        st.write(f"Spam Probability: {spam_conf:.4f}")
        st.write(f"Not Spam Probability: {ham_conf:.4f}")

        st.progress(float(max(spam_conf, ham_conf)))

st.sidebar.title("Model Information")
st.sidebar.write("Algorithm: Multinomial Naive Bayes")
st.sidebar.write("Feature Extraction: TF-IDF")
st.sidebar.write("Task: Binary Text Classification")

st.sidebar.divider()

st.sidebar.title("Notes")
st.sidebar.write(
    "This model is trained on labeled SMS/email data and is optimized for spam detection tasks."
)