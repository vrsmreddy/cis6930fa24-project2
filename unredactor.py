import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score
)
import joblib
import sys
from Levenshtein import distance as levenshtein_distance
import numpy as np

# Ensure required NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# Constants for block character and data file paths
BLOCK_CHAR = '\u2588'
TRAIN_DATA_FILE = 'data/unredactor.tsv'
TEST_DATA_FILE = 'test.tsv'
SUBMISSION_FILE = 'submission.tsv'

def extract_features(context):
    """Extracts features from the context surrounding the redacted name."""
    features = {}
    # Regex to find the block of redacted characters
    redaction_pattern = r'[' + BLOCK_CHAR + r']+'
    match = re.search(redaction_pattern, context)
    if match:
        start, end = match.span()
        before = context[:start].strip()
        after = context[end:].strip()
        before_tokens = word_tokenize(before)
        after_tokens = word_tokenize(after)
        # Extract features from words before and after the redacted name
        for i in range(1, 4):
            features[f'prev_word_{i}'] = (
                before_tokens[-i].lower() if len(before_tokens) >= i else ''
            )
            features[f'next_word_{i}'] = (
                after_tokens[i - 1].lower() if len(after_tokens) >= i else ''
            )
        # Include POS tagging for more contextual features
        pos_before = pos_tag(before_tokens)
        pos_after = pos_tag(after_tokens)
        for i in range(1, 4):
            features[f'prev_pos_{i}'] = (
                pos_before[-i][1] if len(pos_before) >= i else ''
            )
            features[f'next_pos_{i}'] = (
                pos_after[i - 1][1] if len(pos_after) >= i else ''
            )
        # Length of the redaction as a feature
        features['redaction_length'] = end - start
    else:
        # If no redaction found, return None
        return None
    return features

def load_and_preprocess_data(filepath, is_training=True):
    """Loads and preprocesses the data."""
    try:
        if is_training:
            data = pd.read_csv(
                filepath,
                sep='\t',
                names=['split', 'name', 'context'],
                on_bad_lines='warn',  # Skip or warn on bad lines
                engine='python',
                quoting=3,  # csv.QUOTE_NONE
            )
        else:
            data = pd.read_csv(
                filepath,
                sep='\t',
                names=['id', 'context'],
                on_bad_lines='warn',
                engine='python',
                quoting=3,
            )
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)

    if data.empty:
        print(f"No data found in {filepath}. Exiting.")
        sys.exit(1)

    # Apply feature extraction to each context
    data['features'] = data['context'].apply(extract_features)
    # Remove entries with no features extracted
    data = data.dropna(subset=['features'])
    return data

def train_model(X, y):
    """Trains a machine learning model and returns it along with the vectorizer."""
    vectorizer = DictVectorizer(sparse=False)
    X_vectorized = vectorizer.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vectorized, y)
    return model, vectorizer

def evaluate_model(model, vectorizer, X, y_true):
    """Evaluates the model's performance on the validation set."""
    X_vectorized = vectorizer.transform(X)
    y_pred = model.predict(X_vectorized)

    # Calculate Levenshtein distances
    distances = [
        levenshtein_distance(a.lower(), b.lower())
        for a, b in zip(y_true, y_pred)
    ]
    average_distance = sum(distances) / len(distances)
    print(f"Average Levenshtein Distance: {average_distance:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    # Calculate and print precision, recall, and F1-score
    precision = precision_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    f1 = f1_score(
        y_true, y_pred, average='weighted', zero_division=0
    )
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")

def make_predictions(model, vectorizer, data):
    """Makes predictions on the test data and returns a DataFrame with the results."""
    X_test = list(data['features'])
    X_vectorized = vectorizer.transform(X_test)
    class_labels = model.classes_

    # Create a mapping from class labels to their lengths (normalized)
    name_lengths = {
        name: len(
            name.replace(' ', '').replace('.', '').replace("'", "").lower()
        )
        for name in class_labels
    }

    predictions = []
    for i, x in enumerate(X_test):
        # Get the redaction length
        redaction_length = x['redaction_length']
        # Filter class labels by length (within +/- 2 characters)
        candidate_names = [
            name
            for name in class_labels
            if abs(name_lengths[name] - redaction_length) <= 2
        ]
        if not candidate_names:
            # If no candidates found, use all class labels
            candidate_names = class_labels
        # Get indices of candidate names
        candidate_indices = [list(class_labels).index(name) for name in candidate_names]
        # Get predicted probabilities
        probs = model.predict_proba([X_vectorized[i]])[0]
        # Select the candidate with the highest probability
        candidate_probs = probs[candidate_indices]
        best_candidate_index = candidate_indices[np.argmax(candidate_probs)]
        predicted_name = class_labels[best_candidate_index]
        predictions.append(predicted_name)

    # Create a DataFrame with 'id' and 'name'
    results = pd.DataFrame({
        'id': data['id'],
        'name': predictions
    })
    return results

def main():
    # Load and preprocess training data
    print("Loading and preprocessing training data...")
    data = load_and_preprocess_data(TRAIN_DATA_FILE)
    if data.empty:
        print("No data loaded. Exiting.")
        sys.exit(1)

    training_data = data[data['split'] == 'training']
    validation_data = data[data['split'] == 'validation']

    if training_data.empty:
        print("No training data found. Exiting.")
        sys.exit(1)
    if validation_data.empty:
        print("No validation data found. Continuing without validation.")

    # Prepare data for training
    X_train = list(training_data['features'])
    y_train = training_data['name']

    # Train model
    print("Training the model...")
    model, vectorizer = train_model(X_train, y_train)

    # Evaluate model if validation data is available
    if not validation_data.empty:
        # Prepare validation data
        X_validation = list(validation_data['features'])
        y_validation = validation_data['name']

        print("\nEvaluating the model on validation data...")
        evaluate_model(model, vectorizer, X_validation, y_validation)
    else:
        print("No validation data available. Skipping evaluation.")

    # Save the trained model and vectorizer for later use
    joblib.dump(model, 'trained_unredactor_model.pkl')
    joblib.dump(vectorizer, 'feature_vectorizer.pkl')
    print("\nModel and vectorizer saved.")

    # Load and preprocess test data
    print("\nLoading and preprocessing test data...")
    test_data = load_and_preprocess_data(TEST_DATA_FILE, is_training=False)
    if test_data.empty:
        print("No test data loaded. Exiting.")
        sys.exit(1)

    # Make predictions on test data
    print("Making predictions on test data...")
    predictions = make_predictions(model, vectorizer, test_data)

    # Save predictions to submission file
    predictions.to_csv(SUBMISSION_FILE, sep='\t', index=False)
    print(f"Predictions saved to {SUBMISSION_FILE}.")

if __name__ == '__main__':
    main()
