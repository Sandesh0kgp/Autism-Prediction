# 🧠 Autism Spectrum Disorder (ASD) Screening App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://autism-prediction-acovvfdx5bwnugzfpc5gqd.streamlit.app/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Groq AI](https://img.shields.io/badge/AI-Groq-orange.svg)](https://groq.com/)
[![Perplexity AI](https://img.shields.io/badge/Search-Perplexity-purple.svg)](https://perplexity.ai/)

A professional machine learning application for preliminary autism screening, featuring a Random Forest classifier with 93% cross-validation accuracy and an AI-powered chatbot with real-time web search capabilities.

---

## 🌟 Key Features

- **🚀 ML-Powered Screening**: Instant autism trait detection using an optimized Random Forest model trained on clinical AQ-10 screening questions
- **💬 Dual-AI Chatbot**: Context-aware assistant powered by Groq (Llama 3.3 70B) with web-enhanced responses via Perplexity AI
- **🔍 Real-Time Research**: Access to latest autism research and medical information through Perplexity's web search integration
- **📊 Complete History Tracking**: SQLite database stores user profiles, prediction results, and full conversation history
- **👤 Profile Management**: UUID-based user authentication with isolated sessions for data privacy
- **🎨 Modern UI**: Clean, responsive multi-page Streamlit dashboard

---

## 🤖 Machine Learning Model

### Model Architecture:
- **Algorithm**: Random Forest Classifier (50 estimators, max_depth=20, bootstrap=False)
- **Optimization**: RandomizedSearchCV with 5-fold cross-validation over 20 parameter combinations
- **Training Data**: 800 screening records across diverse demographics

### Data Processing Pipeline:
1. **Outlier Treatment**: IQR-based median replacement for `age` and `result` features
2. **Class Balancing**: SMOTE (Synthetic Minority Oversampling) to address 515:285 class imbalance
3. **Feature Engineering**: Label encoding for 7 categorical variables (gender, ethnicity, country, etc.)
4. **Train/Test Split**: 80/20 split with stratification (640 training, 160 testing samples)

### Performance Metrics:
- **Cross-Validation Accuracy**: 93% (5-fold CV on training set)
- **Test Set Accuracy**: 82% on unseen data
- **Precision (ASD Class)**: 59%
- **Recall (ASD Class)**: 64%
- **F1-Score**: 0.61 for minority class, 0.88 for majority class

### Input Features (17 total):
- **A1-A10 Scores**: Binary responses (0/1) to 10 screening questions based on AQ-10 assessment
- **Demographics**: Age, gender, ethnicity, country of residence
- **Medical History**: Jaundice at birth, family history of autism, previous screening experience
- **Computed Feature**: Result score (sum of A-scores)

---

## 🛠️ Tech Stack

### Core Technologies:
- **Frontend**: [Streamlit](https://streamlit.io/) - Multi-page web application
- **ML Framework**: [Scikit-learn 1.6.1](https://scikit-learn.org/) - Random Forest, preprocessing
- **Boosting**: [XGBoost](https://xgboost.readthedocs.io/) - Evaluated during model selection
- **Data Processing**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Imbalanced Learning**: [Imbalanced-learn](https://imbalanced-learn.org/) - SMOTE implementation
- **Visualization**: [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

### AI & APIs:
- **Conversational AI**: [Groq Cloud API](https://console.groq.com/) - Llama 3.3 70B Versatile
- **Web Search**: [Perplexity AI](https://www.perplexity.ai/) - Sonar model with online search
- **Database**: [SQLite3](https://www.sqlite.org/) - Local persistent storage

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Groq API key (get from [console.groq.com](https://console.groq.com/keys))
- Perplexity API key (get from [perplexity.ai/settings/api](https://www.perplexity.ai/settings/api))

### 1. Clone Repository
```bash
git clone https://github.com/Sandesh0kgp/Autism-Prediction.git
cd Autism-Prediction
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
PERPLEXITY_API_KEY=pplx_your_perplexity_api_key_here
DEBUG_MODE=false
```

### 5. Run the Application

```bash
# Default port (8501)
streamlit run app.py

# Custom port (if 8501 is blocked)
streamlit run app.py --server.port 8000
```

The app will open at `http://localhost:8501` (or your custom port).

---

## 📖 How to Use

### 1. Create Your Profile
- Navigate to **👤 Create Profile** in the sidebar
- Enter your name and age
- Receive a unique UUID for session management

### 2. Take the Screening Assessment
- Go to **🔮 Predict**
- Answer 10 binary screening questions (A1-A10 from AQ-10 assessment)
- Provide demographic information (age, gender, ethnicity, country, medical history)
- Get instant prediction with confidence scores

### 3. Chat with AI Assistant
- Navigate to **💬 Chatbot**
- Ask questions about autism, your results, or latest research
- The bot uses your profile context and can search the web for current information
- View cited sources in expandable sections

### 4. Review Your History
- Go to **📊 History**
- See all past predictions with input data
- Review complete chat conversation history

---

## 🏗️ Project Structure

```
Autism-Prediction/
├── app.py                                      # Main Streamlit application
├── chatbot.py                                  # AI chatbot with Groq & Perplexity integration
├── database.py                                 # SQLite operations (CRUD functions)
├── model_utils.py                              # Model loading, preprocessing, prediction
├── requirements.txt                            # Python dependencies
├── .env                                        # API keys (create this locally)
├── .gitignore                                  # Excludes .env, .db, and cache files
├── README.md                                   # This file
├── best_model.pkl                              # Trained Random Forest model
├── encoders.pkl                                # Label encoders for categorical features
├── Autism_Preidiction_using_machine_Learning.ipynb  # Full training notebook
├── train (1).csv                               # Training dataset (800 records)
└── autism.db                                   # SQLite database (auto-created)
```

---

## 🗄️ Database Schema

### `users` Table
| Column      | Type    | Description                    |
|-------------|---------|--------------------------------|
| user_id     | TEXT    | Primary Key (UUID)             |
| name        | TEXT    | User's name                    |
| age         | INTEGER | User's age                     |
| created_at  | TEXT    | ISO timestamp                  |

### `history` Table
| Column        | Type    | Description                         |
|---------------|---------|-------------------------------------|
| id            | INTEGER | Auto-increment primary key          |
| user_id       | TEXT    | Foreign key to users                |
| input_data    | TEXT    | JSON of prediction inputs           |
| prediction    | TEXT    | Prediction result                   |
| user_question | TEXT    | Chat question (nullable)            |
| bot_response  | TEXT    | Chat response (nullable)            |
| timestamp     | TEXT    | ISO timestamp                       |

---

## 🧪 Model Training Process

The model was developed in `Autism_Preidiction_using_machine_Learning.ipynb` following these steps:

1. **Data Exploration**: Analysis of 800 records with 22 original features
2. **Data Cleaning**: Removed ID and redundant age_desc columns; standardized country names
3. **EDA**: Distribution analysis, outlier detection via IQR, correlation heatmaps
4. **Preprocessing**: 
   - Outlier capping with median replacement
   - Label encoding for 7 categorical features
   - Feature scaling not required (tree-based models)
5. **Train/Test Split**: 80/20 stratified split
6. **SMOTE**: Balanced classes from 515/125 to 515/515 in training set
7. **Model Comparison**: Cross-validated Decision Tree (86%), Random Forest (92%), XGBoost (90%)
8. **Hyperparameter Tuning**: RandomizedSearchCV on Random Forest
9. **Final Model**: Random Forest with 93% CV accuracy, 82% test accuracy
10. **Serialization**: Saved model and encoders as `.pkl` files

---

## 🔒 Security & Privacy

- API keys stored in environment variables (never committed to Git)
- `.gitignore` excludes `.env`, database files, and model pickles from version control
- Each user has isolated data accessible only via their UUID
- SQLite database stored locally (no external data transmission)
- Streamlit Cloud deployment uses encrypted secrets management

---

## 🚀 Deployment (Streamlit Cloud)

### Steps:
1. Push code to GitHub (API keys excluded via `.gitignore`)
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Connect your GitHub repository
4. Set **Main file path** to `app.py`
5. Add secrets in **Settings → Secrets**:
```toml
GROQ_API_KEY = "gsk_your_key"
PERPLEXITY_API_KEY = "pplx_your_key"
DEBUG_MODE = "false"
```
6. Deploy and share your live app!

---

## ⚠️ Medical Disclaimer

**This is a screening tool, NOT a diagnostic instrument.**

The predictions are based on statistical patterns from machine learning and should be used for informational purposes only. Autism Spectrum Disorder diagnosis requires comprehensive clinical evaluation by qualified healthcare professionals, including:
- Licensed psychologists
- Developmental pediatricians
- Psychiatrists specializing in ASD

If you or someone you care about shows signs of ASD, please consult a healthcare provider for proper assessment.

---

## 📚 References

- **AQ-10 Screening Tool**: Baron-Cohen et al. (2001) - Autism Spectrum Quotient
- **SMOTE Algorithm**: Chawla et al. (2002) - Synthetic Minority Over-sampling Technique
- **Random Forest**: Breiman (2001) - Random Forests for Classification

---

## 🤝 Contributing

This is a portfolio project demonstrating ML deployment skills. Feel free to fork and adapt for educational purposes.

---

## 📄 License

This project is open-source under the MIT License.

---

**Built with ❤️ for autism awareness and early screening.**

**Live Demo**: [autism-prediction.streamlit.app](https://autism-prediction-acovvfdx5bwnugzfpc5gqd.streamlit.app/)
