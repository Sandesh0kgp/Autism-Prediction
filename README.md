# 🧠 Autism Spectrum Disorder (ASD) Screening App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-link.streamlit.app/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Groq AI](https://img.shields.io/badge/AI-Groq-orange.svg)](https://groq.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A professional, full-stack machine learning application designed to provide preliminary autism screening. This project integrates a high-performance Random Forest model with a context-aware AI chatbot powered by Groq and real-time web search capabilities.

---

## 🌟 Key Features

- **🚀 Smart Screening**: Get instant results from a trained Machine Learning model based on clinically relevant A1-A10 screening questions.
- **💬 AI Assistant**: A context-aware chatbot that knows your profile and prediction results to provide personalized guidance using Groq Cloud (Llama 3.3 70B).
- **🔍 Web-Enhanced Answers**: Integration with DuckDuckGo to provide the latest research and information beyond the AI's training knowledge.
- **📊 Activity Tracking**: Full history of your screenings and conversations, stored locally and securely in SQLite.
- **👤 Profile Management**: Isolated user sessions with unique IDs for a personalized experience.
- **🎨 Modern UI**: A clean, responsive dashboard built with Streamlit.

---

## 🤖 Model Intelligence

The core of this application is a predictive model trained on a dataset of 800 screening records.

### Model Details:
- **Algorithm**: Random Forest Classifier (Optimized via RandomizedSearchCV)
- **Training Techniques**:
    - **SMOTE**: Used Synthetic Minority Oversampling Technique to handle class imbalance.
    - **Outlier Management**: Applied IQR-based median replacement for numerical features (Age, Result).
    - **Label Encoding**: Categorical features (Gender, Ethnicity, Country, etc.) are processed using pre-trained Label Encoders.
- **Features**: 17 total input features, including 10 screening question scores (A1-A10), demographic data, and medical history.
- **Performance**: Achieved **~93% Cross-Validation Accuracy** during training and **~82% Accuracy** on unseen test data.

---

## 🛠️ Tech Stack

- **Frontend/UI**: [Streamlit](https://streamlit.io/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [Pandas](https://pandas.pydata.org/)
- **AI/LLM**: [Groq Cloud API](https://console.groq.com/) (Llama 3.3 70B)
- **Database**: [SQLite](https://www.sqlite.org/)
- **Search API**: [DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)

---

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/autism-screening-app.git
cd autism-screening-app

# Create and activate virtual environment
python -m venv venv
# On Windows use: 
.\venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```env
GROQ_API_KEY=your_groq_api_key_here
DEBUG_MODE=false
```

### 3. Running the App

```bash
# Standard run
streamlit run app.py

# If port 8501 is blocked (common on Windows), use:
streamlit run app.py --server.port 8000
```

---

## ⚠️ Medical Disclaimer

**This application is a screening tool, not a diagnostic tool.** 
The results provided are based on machine learning patterns and should be used for informational purposes only. If you or someone you know is concerned about ASD, please consult a qualified healthcare professional or specialist for a formal clinical evaluation.

---

**Built with ❤️ for better ASD awareness and early screening.**
