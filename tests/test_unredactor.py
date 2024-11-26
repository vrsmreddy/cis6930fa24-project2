import pytest
import sys
import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import joblib

# Import the functions from unredactor.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from unredactor import (
    extract_features,
    load_and_preprocess_data,
    train_model,
    evaluate_model,
    make_predictions
)

# Constants
BLOCK_CHAR = '\u2588'

# Sample data for testing
sample_context = "I had a meeting with " + BLOCK_CHAR * 5 + " yesterday."
expected_features = {
    'prev_word_1': 'with',
    'next_word_1': 'yesterday',
    'prev_pos_1': 'IN',
    'next_pos_1': 'NN',
    'redaction_length': 5
}

def test_extract_features():
    """Test the feature extraction function."""
    features = extract_features(sample_context)
    assert features is not None, "Features should not be None"
    assert features['prev_word_1'] == expected_features['prev_word_1'], "Incorrect previous word"
    assert features['next_word_1'] == expected_features['next_word_1'], "Incorrect next word"
    assert features['prev_pos_1'] == expected_features['prev_pos_1'], "Incorrect previous POS tag"
    assert features['next_pos_1'] == expected_features['next_pos_1'], "Incorrect next POS tag"
    assert features['redaction_length'] == expected_features['redaction_length'], "Incorrect redaction length"

def test_load_and_preprocess_data():
    """Test data loading and preprocessing."""
    # Create a small DataFrame to simulate the data
    data = pd.DataFrame({
        'split': ['training', 'validation'],
        'name': ['John Doe', 'Jane Smith'],
        'context': [sample_context, sample_context]
    })
    # Save to a temporary CSV file
    temp_file = 'temp_unredactor.tsv'
    data.to_csv(temp_file, sep='\t', index=False, header=False)

    # Load the data using the function
    loaded_data = load_and_preprocess_data(temp_file)
    os.remove(temp_file)  # Clean up

    assert not loaded_data.empty, "Loaded data should not be empty"
    assert 'features' in loaded_data.columns, "Features column should exist"
    assert len(loaded_data) == 2, "Loaded data should have 2 entries"

def test_train_model():
    """Test model training."""
    # Create sample feature dictionaries
    X = [
        {'prev_word_1': 'with', 'next_word_1': 'yesterday', 'redaction_length': 5},
        {'prev_word_1': 'with', 'next_word_1': 'tomorrow', 'redaction_length': 7}
    ]
    y = ['John Doe', 'Jane Smith']

    model, vectorizer = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier"
    assert isinstance(vectorizer, DictVectorizer), "Vectorizer should be a DictVectorizer"

def test_evaluate_model(capsys):
    """Test model evaluation."""
    # Use the same data as in train_model test
    X = [
        {'prev_word_1': 'with', 'next_word_1': 'yesterday', 'redaction_length': 5},
        {'prev_word_1': 'with', 'next_word_1': 'tomorrow', 'redaction_length': 7}
    ]
    y = ['John Doe', 'Jane Smith']
    model, vectorizer = train_model(X, y)

    # Evaluate the model
    evaluate_model(model, vectorizer, X, y)
    captured = capsys.readouterr()
    assert "Classification Report:" in captured.out, "Should print classification report"

def test_make_predictions():
    """Test making predictions."""
    # Train a simple model
    X_train = [
        {'prev_word_1': 'with', 'next_word_1': 'yesterday', 'redaction_length': 5},
        {'prev_word_1': 'with', 'next_word_1': 'tomorrow', 'redaction_length': 7}
    ]
    y_train = ['John Doe', 'Jane Smith']
    model, vectorizer = train_model(X_train, y_train)

    # Create test data
    test_data = pd.DataFrame({
        'id': [1, 2],
        'features': X_train
    })

    # Make predictions
    results = make_predictions(model, vectorizer, test_data)
    assert 'id' in results.columns, "Results should contain 'id' column"
    assert 'name' in results.columns, "Results should contain 'name' column"
    assert len(results) == 2, "Results should have 2 entries"

def test_full_pipeline(tmpdir):
    """Test the full pipeline from data loading to prediction."""
    # Create a small dataset
    train_data = pd.DataFrame({
        'split': ['training', 'training'],
        'name': ['John Doe', 'Jane Smith'],
        'context': [
            "I met with " + BLOCK_CHAR * 8 + " yesterday.",
            "I spoke to " + BLOCK_CHAR * 10 + " today."
        ]
    })
    test_data = pd.DataFrame({
        'id': [1],
        'context': ["I met with " + BLOCK_CHAR * 8 + " yesterday."]
    })

    # Save to temporary files
    train_file = tmpdir.join('temp_unredactor.tsv')
    test_file = tmpdir.join('temp_test.tsv')
    train_data.to_csv(train_file, sep='\t', index=False, header=False)
    test_data.to_csv(test_file, sep='\t', index=False, header=False)

    # Load and preprocess data
    training_data = load_and_preprocess_data(str(train_file))
    X_train = list(training_data['features'])
    y_train = training_data['name']

    # Train model
    model, vectorizer = train_model(X_train, y_train)

    # Load and preprocess test data
    test_data_loaded = load_and_preprocess_data(str(test_file), is_training=False)

    # Make predictions
    predictions = make_predictions(model, vectorizer, test_data_loaded)
    assert len(predictions) == 1, "Should have one prediction"
    assert predictions.iloc[0]['id'] == 1, "ID should be 1"
    assert predictions.iloc[0]['name'] in ['John Doe', 'Jane Smith'], "Name should be one of the trained names"
