
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from utils.text_processor import TextProcessor

def load_liar_dataset(dataset_path='C:/Users/Gogul/Downloads/liar_dataset'):
    """Load and process LIAR dataset"""
    print("Loading LIAR dataset...")
    
    # LIAR dataset files
    train_file = os.path.join(dataset_path, 'train.tsv')
    test_file = os.path.join(dataset_path, 'test.tsv')
    valid_file = os.path.join(dataset_path, 'valid.tsv')
    
    # Column names for LIAR dataset
    columns = [
        'id', 'label', 'statement', 'subject', 'speaker', 'speaker_job', 
        'state_info', 'party_affiliation', 'barely_true_counts', 'false_counts',
        'half_true_counts', 'mostly_true_counts', 'pants_on_fire_counts', 'context'
    ]
    
    dfs = []
    
    # Load train file
    if os.path.exists(train_file):
        df_train = pd.read_csv(train_file, sep='\t', header=None, names=columns)
        dfs.append(df_train)
        print(f"Loaded train set: {len(df_train)} samples")
    
    # Load test file
    if os.path.exists(test_file):
        df_test = pd.read_csv(test_file, sep='\t', header=None, names=columns)
        dfs.append(df_test)
        print(f"Loaded test set: {len(df_test)} samples")
    
    # Load validation file
    if os.path.exists(valid_file):
        df_valid = pd.read_csv(valid_file, sep='\t', header=None, names=columns)
        dfs.append(df_valid)
        print(f"Loaded validation set: {len(df_valid)} samples")
    
    if not dfs:
        raise FileNotFoundError(f"No LIAR dataset files found in {dataset_path}")
    
    # Combine all datasets
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total combined dataset: {len(df)} samples")
    
    # Process labels - convert LIAR 6-class to binary
    # LIAR labels: 'false', 'mostly-false', 'half-true', 'mostly-true', 'true', 'pants-on-fire'
    # Convert to binary: 0 = Fake, 1 = Real
    fake_labels = ['false', 'mostly-false', 'pants-on-fire']
    real_labels = ['true', 'mostly-true', 'half-true']
    
    df['binary_label'] = df['label'].apply(
        lambda x: 0 if x in fake_labels else (1 if x in real_labels else None)
    )
    
    # Remove rows with None labels (shouldn't happen with proper LIAR dataset)
    df = df.dropna(subset=['binary_label'])
    
    # Use statement column as text
    df['text'] = df['statement'].fillna('')
    
    print(f"Label distribution:")
    print(df['binary_label'].value_counts())
    print(f"Original labels distribution:")
    print(df['label'].value_counts())
    
    return df[['text', 'binary_label']].rename(columns={'binary_label': 'label'})

def train_models_with_liar():
    """Train models with LIAR dataset"""
    try:
        # Load LIAR dataset
        df = load_liar_dataset()
        
        # Process text
        processor = TextProcessor()
        print("Processing text...")
        df['processed_text'] = df['text'].apply(processor.process)
        
        # Remove empty processed texts
        df = df[df['processed_text'].str.len() > 0]
        print(f"Dataset after text processing: {len(df)} samples")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], 
            df['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=df['label']  # Maintain label distribution
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Vectorize
        print("Vectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=10000,  # Increased for better performance
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
            sublinear_tf=True  # Apply sublinear tf scaling
        )
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        # Train models
        models = {}
        
        print("\nTraining Logistic Regression...")
        lr = LogisticRegression(
            max_iter=2000, 
            random_state=42,
            class_weight='balanced'  # Handle class imbalance
        )
        lr.fit(X_train_vec, y_train)
        models['logistic_regression'] = lr
        y_pred_lr = lr.predict(X_test_vec)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_lr, target_names=['Fake', 'Real']))
        
        print("\nTraining Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200, 
            random_state=42,
            class_weight='balanced',
            max_depth=50,
            min_samples_split=5
        )
        rf.fit(X_train_vec, y_train)
        models['random_forest'] = rf
        y_pred_rf = rf.predict(X_test_vec)
        acc_rf = accuracy_score(y_test, y_pred_rf)
        print(f"Random Forest Accuracy: {acc_rf:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_rf, target_names=['Fake', 'Real']))
        
        print("\nTraining Naive Bayes...")
        nb = MultinomialNB(alpha=1.0)
        nb.fit(X_train_vec, y_train)
        models['naive_bayes'] = nb
        y_pred_nb = nb.predict(X_test_vec)
        acc_nb = accuracy_score(y_test, y_pred_nb)
        print(f"Naive Bayes Accuracy: {acc_nb:.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred_nb, target_names=['Fake', 'Real']))
        
        # Save models
        print("\nSaving models...")
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'vectorizer': vectorizer,
            'models': models,
            'dataset_info': {
                'name': 'LIAR',
                'samples': len(df),
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'accuracies': {
                    'logistic_regression': acc_lr,
                    'random_forest': acc_rf,
                    'naive_bayes': acc_nb
                }
            }
        }
        
        joblib.dump(model_data, 'models/trained_model.pkl')
        
        print("Models trained and saved successfully!")
        print(f"\nModel Performance Summary:")
        print(f"Logistic Regression: {acc_lr:.4f}")
        print(f"Random Forest: {acc_rf:.4f}")
        print(f"Naive Bayes: {acc_nb:.4f}")
        
        return model_data
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure the LIAR dataset is available at C:/Users/Gogul/Downloads/liar_dataset/")
        print("Required files: train.tsv, test.tsv, valid.tsv")
        return None
    except Exception as e:
        print(f"Error training models: {e}")
        return None

if __name__ == '__main__':
    train_models_with_liar()