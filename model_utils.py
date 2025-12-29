"""
Model utilities for loading and running autism prediction model.
Handles model loading, input preprocessing, and predictions.
"""

import pickle
import pandas as pd
import numpy as np
from typing import Dict, Tuple
import os


MODEL_PATH = "best_model.pkl"
ENCODERS_PATH = "encoders.pkl"


def load_model() -> Tuple:
    """
    Load the trained model and encoders.
    
    Returns:
        Tuple of (model, encoders)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Please run your Jupyter notebook (Autism_Preidiction_using_machine_Learning.ipynb) "
            "to train and save the model."
        )
    
    if not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError(
            f"Encoders file '{ENCODERS_PATH}' not found. "
            "Please run your Jupyter notebook (Autism_Preidiction_using_machine_Learning.ipynb) "
            "to train and save the encoders."
        )
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(ENCODERS_PATH, 'rb') as f:
        encoders = pickle.load(f)
    
    return model, encoders


def preprocess_input(input_data: Dict, encoders: Dict) -> pd.DataFrame:
    """
    Preprocess user input to match model training format.
    
    Args:
        input_data: Dictionary with user inputs
        encoders: Dictionary of label encoders
        
    Returns:
        DataFrame ready for prediction
    """
    # Create a copy to avoid modifying original
    data = input_data.copy()
    
    # Categorical columns that need encoding (must match training)
    categorical_cols = ['gender', 'ethnicity', 'jaundice', 'austim', 
                       'contry_of_res', 'used_app_before', 'relation']
    
    # Encode categorical variables
    for col in categorical_cols:
        if col in encoders and col in data:
            try:
                # Handle unknown categories by using the first category
                if data[col] not in encoders[col].classes_:
                    data[col] = encoders[col].classes_[0]
                data[col] = encoders[col].transform([data[col]])[0]
            except Exception as e:
                print(f"Warning: Error encoding {col}: {e}")
                data[col] = 0
    
    # Add 'result' column (calculated feature from notebook)
    # This is a computed score based on A-scores - using sum as approximation
    if 'result' not in data:
        a_scores_sum = sum([data.get(f'A{i}_Score', 0) for i in range(1, 11)])
        # Simple approximation - in real notebook this is more complex
        data['result'] = float(a_scores_sum)
    
    # Add 'relation' if not present (default to 'Self')
    if 'relation' not in data:
        data['relation'] = 'Self'
        if 'relation' in encoders:
            try:
                data['relation'] = encoders['relation'].transform(['Self'])[0]
            except:
                data['relation'] = 0
    
    # Create DataFrame with correct column order (must match training)
    # Based on notebook: all columns except 'Class/ASD', 'ID', 'age_desc'
    columns = [
        'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
        'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score',
        'age', 'gender', 'ethnicity', 'jaundice', 'austim',
        'contry_of_res', 'used_app_before', 'result', 'relation'
    ]
    
    # Ensure all columns are present
    for col in columns:
        if col not in data:
            data[col] = 0
    
    # Create DataFrame
    df = pd.DataFrame([data], columns=columns)
    
    return df


def predict(input_data: Dict) -> Dict:
    """
    Make a prediction using the trained model.
    
    Args:
        input_data: Dictionary with user inputs
        
    Returns:
        Dictionary with prediction result and probability
    """
    try:
        # Load model and encoders
        model, encoders = load_model()
        
        # Preprocess input
        X = preprocess_input(input_data, encoders)
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        # Get probability if available
        try:
            probability = model.predict_proba(X)[0]
            prob_dict = {
                'no_autism': float(probability[0]),
                'autism': float(probability[1])
            }
        except:
            prob_dict = None
        
        # Interpret result
        result = "Autism Spectrum Disorder (ASD) traits detected" if prediction == 1 else "No ASD traits detected"
        
        return {
            'prediction': int(prediction),
            'result': result,
            'probabilities': prob_dict
        }
        
    except Exception as e:
        return {
            'error': str(e),
            'prediction': None,
            'result': f"Error making prediction: {str(e)}"
        }


def get_feature_names() -> Dict:
    """
    Get feature names and their descriptions for the UI.
    
    Returns:
        Dictionary mapping feature names to descriptions
    """
    return {
        'A1_Score': 'A1: I often notice small sounds when others do not',
        'A2_Score': 'A2: I usually concentrate more on the whole picture, rather than small details',
        'A3_Score': 'A3: I find it easy to do more than one thing at once',
        'A4_Score': 'A4: If there is an interruption, I can switch back very quickly',
        'A5_Score': 'A5: I find it easy to read between the lines',
        'A6_Score': 'A6: I know how to tell if someone is interested or bored',
        'A7_Score': 'A7: I find it easy to work out what someone is thinking or feeling',
        'A8_Score': 'A8: I find it difficult to work out people\'s intentions',
        'A9_Score': 'A9: I enjoy social occasions',
        'A10_Score': 'A10: I find it difficult to work out what to do in a social situation',
        'age': 'Age (in years)',
        'gender': 'Gender (m/f)',
        'ethnicity': 'Ethnicity',
        'jaundice': 'Born with jaundice? (yes/no)',
        'austim': 'Family member with autism? (yes/no)',
        'contry_of_res': 'Country of residence',
        'used_app_before': 'Used screening app before? (yes/no)'
    }
