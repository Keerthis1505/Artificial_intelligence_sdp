import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Emotion Detection AI",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎭 Emotion Detection AI")
st.markdown("Train and predict emotions from text using Machine Learning")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🏠 Home", "🔮 Predict Emotion", "📊 Model Performance", "🔧 Retrain Model"])

# Load model and vectorizer
@st.cache_resource
def load_model():
    try:
        model = joblib.load('emotion_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        return model, vectorizer
    except:
        st.error("Model files not found. Please train the model first.")
        return None, None

model, vectorizer = load_model()

# =============== HOME PAGE ===============
if page == "🏠 Home":
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Welcome! 👋")
        st.markdown("""
        ### About This App
        
        This application uses **Machine Learning** to detect emotions from text input.
        
        **Emotions Detected:**
        - 😊 **Joy**
        - 😢 **Sadness**
        - 😡 **Anger**
        - 😨 **Fear**
        - 💕 **Love**
        - 😮 **Surprise**
        
        ### How it Works
        1. Enter text
        2. Model analyzes the text
        3. Predicts the emotion with confidence score
        """)
    
    with col2:
        st.header("📈 Model Stats")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Overall Accuracy", "91%", "+6%")
            st.metric("Joy Accuracy", "97.65%", "🏆")
        with col_b:
            st.metric("Sadness Accuracy", "96.81%", "🏆")
            st.metric("Training Samples", "16,000", "📚")
    
    st.markdown("---")
    st.header("🚀 Quick Start")
    
    quick_text = st.text_area(
        "Try it now! Enter text to detect emotion:",
        placeholder="Example: I feel so happy today!",
        height=100
    )
    
    if st.button("🎯 Detect Emotion", key="home_predict"):
        if quick_text and model and vectorizer:
            vec = vectorizer.transform([quick_text])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            
            emotion_emojis = {
                'joy': '😊', 'sadness': '😢', 'anger': '😡',
                'fear': '😨', 'love': '💕', 'surprise': '😮'
            }
            
            st.success(f"Detected Emotion: **{emotion_emojis.get(pred, '')} {pred.upper()}**")
            
            # Confidence bars
            st.markdown("### Confidence Scores")
            for emotion, score in zip(model.classes_, proba):
                st.write(f"{emotion.capitalize()}: {score:.2%}")
                st.progress(score)

# =============== PREDICTION PAGE ===============
elif page == "🔮 Predict Emotion":
    st.header("Emotion Prediction")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_text = st.text_area(
            "Enter text to analyze:",
            placeholder="Type or paste any text here...",
            height=150
        )
    
    with col2:
        st.markdown("### Tips")
        st.info("""
        - Use natural language
        - Longer text = better results
        - Clear emotional words help
        """)
    
    if st.button("🔍 Predict Now", key="predict_main"):
        if user_text and model and vectorizer:
            vec = vectorizer.transform([user_text])
            pred = model.predict(vec)[0]
            proba = model.predict_proba(vec)[0]
            
            emotion_emojis = {
                'joy': '😊', 'sadness': '😢', 'anger': '😡',
                'fear': '😨', 'love': '💕', 'surprise': '😮'
            }
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Primary Emotion", pred.upper(), emoji=emotion_emojis.get(pred))
            with col2:
                confidence = max(proba) * 100
                st.metric("Confidence", f"{confidence:.1f}%", delta=f"{confidence-50:.1f}%")
            with col3:
                st.metric("Text Length", len(user_text.split()), "words")
            
            st.markdown("---")
            
            # Confidence breakdown
            st.subheader("Emotion Probabilities")
            
            df_proba = pd.DataFrame({
                'Emotion': model.classes_,
                'Confidence': proba
            }).sort_values('Confidence', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#FF6B6B' if e == pred else '#4ECDC4' for e in df_proba['Emotion']]
            ax.barh(df_proba['Emotion'], df_proba['Confidence'], color=colors)
            ax.set_xlabel('Confidence Score')
            ax.set_title('Emotion Detection Results')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show probabilities
            st.markdown("### Detailed Scores")
            for idx, row in df_proba.iterrows():
                st.write(f"**{row['Emotion'].capitalize()}**: {row['Confidence']:.2%}")

# =============== MODEL PERFORMANCE PAGE ===============
elif page == "📊 Model Performance":
    st.header("Model Performance Metrics")
    
    # Load dataset for evaluation
    @st.cache_data
    def load_data():
        data = []
        with open('train.txt', 'r') as f:
            for line in f:
                if ';' in line:
                    text, label = line.rsplit(';', 1)
                    data.append({'text': text.strip(), 'label': label.strip()})
        return pd.DataFrame(data)
    
    df = load_data()
    
    if model and vectorizer:
        X = df['text']
        y = df['label']
        
        X_vec = vectorizer.transform(X)
        y_pred = model.predict(X_vec)
        
        # Overall metrics
        col1, col2, col3, col4 = st.columns(4)
        
        accuracy = (y_pred == y).sum() / len(y)
        with col1:
            st.metric("Overall Accuracy", f"{accuracy*100:.2f}%")
        
        with col2:
            st.metric("Total Samples", len(y))
        
        with col3:
            st.metric("Emotions Detected", len(model.classes_))
        
        with col4:
            st.metric("Features Used", 5000)
        
        st.markdown("---")
        
        # Accuracy per emotion
        st.subheader("📈 Accuracy by Emotion")
        
        accuracy_data = []
        for emotion in model.classes_:
            mask = y == emotion
            acc = (y_pred[mask] == y[mask]).sum() / mask.sum()
            accuracy_data.append({'Emotion': emotion.capitalize(), 'Accuracy': acc})
        
        df_acc = pd.DataFrame(accuracy_data).sort_values('Accuracy', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ECC71' if x >= 0.90 else '#F39C12' if x >= 0.80 else '#E74C3C' for x in df_acc['Accuracy']]
        ax.bar(df_acc['Emotion'], df_acc['Accuracy'], color=colors)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy by Emotion')
        ax.set_ylim([0, 1])
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Confusion Matrix
        st.subheader("🔲 Confusion Matrix")
        
        cm = confusion_matrix(y, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=model.classes_, 
                   yticklabels=model.classes_,
                   cbar_kws={'label': 'Number of Samples'})
        ax.set_ylabel('True Emotion')
        ax.set_xlabel('Predicted Emotion')
        ax.set_title('Confusion Matrix')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Classification Report
        st.subheader("📋 Classification Report")
        
        report = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(3))

# =============== RETRAIN PAGE ===============
elif page == "🔧 Retrain Model":
    st.header("Retrain Model")
    
    st.info("⚠️ Retraining will update the model with current data from train.txt")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Retraining Options
        
        **Model**: Logistic Regression  
        **Vectorizer**: TF-IDF (5000 features)  
        **Test Size**: 20% of data  
        
        Click the button to retrain:
        """)
    
    with col2:
        if st.button("🚀 Retrain Model Now", key="retrain"):
            with st.spinner('Training in progress...'):
                # Load data
                data = []
                with open('train.txt', 'r') as f:
                    for line in f:
                        if ';' in line:
                            text, label = line.rsplit(';', 1)
                            data.append({'text': text.strip(), 'label': label.strip()})
                
                df_train = pd.DataFrame(data)
                
                # Split data
                X = df_train['text']
                y = df_train['label']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                # Vectorize
                vect = TfidfVectorizer(max_features=5000)
                X_train_vec = vect.fit_transform(X_train)
                X_test_vec = vect.transform(X_test)
                
                # Train
                mdl = LogisticRegression(max_iter=1000)
                mdl.fit(X_train_vec, y_train)
                
                # Save
                joblib.dump(mdl, 'emotion_model.pkl')
                joblib.dump(vect, 'vectorizer.pkl')
                
                # Evaluate
                y_pred = mdl.predict(X_test_vec)
                accuracy = (y_pred == y_test).sum() / len(y_test)
                
                st.success("✅ Model trained successfully!")
                st.metric("Test Accuracy", f"{accuracy*100:.2f}%")
                
                # Show classification report
                st.subheader("Test Results")
                from sklearn.metrics import classification_report
                report = classification_report(y_test, y_pred)
                st.text(report)

st.markdown("---")
st.markdown("""
<div style="text-align: center">
<p>Made with ❤️ using Streamlit | Emotion Detection AI v1.0</p>
</div>
""", unsafe_allow_html=True)
