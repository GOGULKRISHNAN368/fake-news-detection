import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

class FakeNewsDetector:
    def __init__(self):
        self.vectorizer = None
        self.models = {
            'logistic_regression': None,
            'random_forest': None,
            'naive_bayes': None
        }
        self.dataset_info = None
        self.load_or_create_models()
    
    def load_or_create_models(self):
        """Load existing models or create new ones"""
        model_path = 'models/trained_model.pkl'
        
        if os.path.exists(model_path):
            try:
                saved_data = joblib.load(model_path)
                self.vectorizer = saved_data['vectorizer']
                self.models = saved_data['models']
                self.dataset_info = saved_data.get('dataset_info', None)
                
                if self.dataset_info:
                    print(f"Models loaded successfully (Dataset: {self.dataset_info.get('name', 'Unknown')})")
                    print(f"Training samples: {self.dataset_info.get('train_samples', 'Unknown')}")
                    if 'accuracies' in self.dataset_info:
                        print("Model accuracies:")
                        for model_name, accuracy in self.dataset_info['accuracies'].items():
                            print(f"  {model_name}: {accuracy:.4f}")
                else:
                    print("Models loaded successfully")
                    
            except Exception as e:
                print(f"Error loading models: {e}")
                self.create_default_models()
        else:
            print("No saved models found. Creating default models...")
            print("To use LIAR dataset, run: python train_model.py")
            self.create_default_models()
    
    def create_default_models(self):
        """Create and train default models with sample data"""
        print("Creating default models with sample data...")
        print("For better accuracy, train with LIAR dataset using train_model.py")
        
        # Enhanced sample training data
        fake_samples = [
            "Breaking: Aliens land in New York City, government covers up massive UFO invasion",
            "Miracle cure discovered that doctors don't want you to know about, cures all diseases instantly",
            "Celebrity dead at 30, shocking details revealed by anonymous insider sources",
            "Scientists prove earth is flat, NASA has been lying all along for decades",
            "New vaccine contains microchips for mind control, government conspiracy exposed",
            "President secretly replaced by body double, shocking evidence revealed",
            "Drinking bleach cures coronavirus, study shows 100 percent effectiveness",
            "5G towers cause cancer and control your mind, experts warn",
            "Local man discovers one weird trick that makes doctors hate him",
            "Government plans to ban oxygen to control population, leaked documents show"
        ]
        
        real_samples = [
            "Stock market shows modest gains amid economic recovery, analysts report steady growth",
            "New study published in Nature journal reveals significant climate change findings",
            "Government announces new infrastructure bill with detailed budget allocation",
            "Research team at university develops improved solar panel technology for renewable energy",
            "International summit addresses global trade policies and economic cooperation",
            "Health officials report decrease in flu cases following vaccination campaign",
            "Technology company releases quarterly earnings report showing revenue growth",
            "Scientists publish peer-reviewed research on marine biology conservation efforts",
            "Local school district implements new educational programs to improve student outcomes",
            "Transportation department announces road maintenance schedule for upcoming months"
        ]
        
        # Create dataset
        texts = fake_samples + real_samples
        labels = [0] * len(fake_samples) + [1] * len(real_samples)
        
        # Create vectorizer and train models with better parameters
        self.vectorizer = TfidfVectorizer(
            max_features=5000, 
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )
        X = self.vectorizer.fit_transform(texts)
        
        # Train multiple models with better parameters
        self.models['logistic_regression'] = LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced'
        )
        self.models['logistic_regression'].fit(X, labels)
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            class_weight='balanced',
            max_depth=10
        )
        self.models['random_forest'].fit(X, labels)
        
        self.models['naive_bayes'] = MultinomialNB(alpha=1.0)
        self.models['naive_bayes'].fit(X, labels)
        
        # Create dataset info
        self.dataset_info = {
            'name': 'Sample Data',
            'samples': len(texts),
            'train_samples': len(texts),
            'test_samples': 0,
            'note': 'Default models with sample data. Train with LIAR dataset for better accuracy.'
        }
        
        # Save models
        self.save_models()
        print("Default models created and saved")
        print("Note: These models use limited sample data. For production use, train with LIAR dataset.")
    
    def save_models(self):
        """Save models to disk"""
        os.makedirs('models', exist_ok=True)
        model_data = {
            'vectorizer': self.vectorizer,
            'models': self.models,
            'dataset_info': self.dataset_info
        }
        joblib.dump(model_data, 'models/trained_model.pkl')
    
    def predict(self, text):
        """Predict if news is fake or real using all models"""
        if not self.vectorizer or not all(self.models.values()):
            raise Exception("Models not loaded")
        
        # Vectorize input
        X = self.vectorizer.transform([text])
        
        # Get predictions from all models
        predictions = {}
        probabilities = []
        
        for name, model in self.models.items():
            pred = model.predict(X)[0]
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[0]
                confidence = float(max(proba))
                probabilities.append(proba)
            else:
                confidence = 0.75  # Default confidence for models without probability
                probabilities.append([0.25, 0.75] if pred == 1 else [0.75, 0.25])
            
            predictions[name] = {
                'label': 'Real' if pred == 1 else 'Fake',
                'confidence': round(confidence * 100, 2),
                'prediction': int(pred)
            }
        
        # Ensemble prediction (weighted average based on model performance)
        if self.dataset_info and 'accuracies' in self.dataset_info:
            # Use model accuracies as weights
            weights = []
            model_names = ['logistic_regression', 'random_forest', 'naive_bayes']
            for model_name in model_names:
                accuracy = self.dataset_info['accuracies'].get(model_name, 0.5)
                weights.append(accuracy)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1/3, 1/3, 1/3]  # Equal weights if no accuracy info
            
            # Weighted average of probabilities
            weighted_proba = np.average(probabilities, axis=0, weights=weights)
        else:
            # Simple average if no accuracy information
            weighted_proba = np.mean(probabilities, axis=0)
        
        ensemble_pred = 1 if weighted_proba[1] > weighted_proba[0] else 0
        ensemble_confidence = float(max(weighted_proba))
        
        predictions['ensemble'] = {
            'label': 'Real' if ensemble_pred == 1 else 'Fake',
            'confidence': round(ensemble_confidence * 100, 2),
            'prediction': int(ensemble_pred)
        }
        
        return predictions
    
    def get_model_info(self):
        """Get information about loaded models"""
        info = {
            'models_loaded': self.is_loaded(),
            'dataset_info': self.dataset_info,
            'vectorizer_features': self.vectorizer.max_features if self.vectorizer else None
        }
        return info
    
    def is_loaded(self):
        """Check if models are loaded"""
        return self.vectorizer is not None and all(self.models.values())