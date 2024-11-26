# Project-2

# Unredactor: 

## Overview

The **Unredactor** project aims to develop a machine learning model capable of reconstructing redacted names in textual documents. Given a context where a name has been redacted, the model predicts the most likely original name based on the surrounding text features.

This README provides comprehensive instructions on how to replicate the pipeline, train the model, evaluate its performance on a validation set, and generate precision, recall, and F1-score metrics. It also outlines the assumptions made, provides examples of usage, and explains how to run tests for each part of the code.

---


## Project Structure

```
├── data/
│   └── unredactor.tsv       # Training and validation data
├── test.tsv                  # Test data
├── submission.tsv            # Output predictions
├── unredactor.py             # Main script for training and prediction
├── tests/
│   └── test_unredactor.py    # Unit tests using pytest
├── Pipfile                   # Pipenv environment setup
├── Pipfile.lock              # Pipenv lockfile
└── README.md                 # Project documentation
```

---

## Installation

### Prerequisites

- Python 3.12.7 or higher
- Pipenv for environment management

### Steps

1. **Clone the Repository**

   ```
   git clone https://github.com/yourusername/unredactor.git
   cd unredactor
   ```

2. **Install Pipenv**

   Ensure you have Pipenv installed:

   ```
   pip install pipenv
   ```

3. **Install Dependencies**

   ```
   pipenv install
   ```

4. **Activate the Virtual Environment**

   ```
   pipenv shell
   ```

---


## Pipeline Description

### Dependencies

The following Python libraries are required for this pipeline:

```python
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib
import sys
from Levenshtein import distance as levenshtein_distance
import numpy as np
```

Ensure that the necessary NLTK resources are downloaded:

```python
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
```

### File Paths

```python
TRAIN_DATA_FILE = 'data/unredactor.tsv'
TEST_DATA_FILE = 'test.tsv'
SUBMISSION_FILE = 'submission.tsv'
```

- `TRAIN_DATA_FILE`: Path to the training dataset.
- `TEST_DATA_FILE`: Path to the test dataset.
- `SUBMISSION_FILE`: Path to save the prediction results.

### Feature Extraction

The core of this pipeline is extracting features from the context surrounding the redacted name:

```python
def extract_features(context):
    features = {}
    redaction_pattern = r'[' + BLOCK_CHAR + r']+'
    match = re.search(redaction_pattern, context)
    if match:
        start, end = match.span()
        before = context[:start].strip()
        after = context[end:].strip()
        before_tokens = word_tokenize(before)
        after_tokens = word_tokenize(after)
        for i in range(1, 4):
            features[f'prev_word_{i}'] = before_tokens[-i].lower() if len(before_tokens) >= i else ''
            features[f'next_word_{i}'] = after_tokens[i-1].lower() if len(after_tokens) >= i else ''
        pos_before = pos_tag(before_tokens)
        pos_after = pos_tag(after_tokens)
        for i in range(1, 4):
            features[f'prev_pos_{i}'] = pos_before[-i][1] if len(pos_before) >= i else ''
            features[f'next_pos_{i}'] = pos_after[i-1][1] if len(pos_after) >= i else ''
        features['redaction_length'] = end - start
    else:
        return None
    return features
```

- **Previous and Next Words**: Extract up to three words before and after the redacted name.
- **Part-of-Speech (POS) Tags**: Extract POS tags for the preceding and following words.
- **Redaction Length**: The length of the redacted block.

### Model Training

The model is trained using a Random Forest Classifier:

```python
def train_model(X, y):
    vectorizer = DictVectorizer(sparse=False)
    X_vectorized = vectorizer.fit_transform(X)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_vectorized, y)
    return model, vectorizer
```

- **Vectorization**: Features are vectorized using `DictVectorizer` to prepare them for the model.
- **Model Training**: The Random Forest model is trained using the function `train_model(X, y)`.

### Evaluation

The model is evaluated using standard classification metrics, along with Levenshtein distance to assess name similarity:

```python
def evaluate_model(model, vectorizer, X, y_true):
    X_vectorized = vectorizer.transform(X)
    y_pred = model.predict(X_vectorized)

    distances = [levenshtein_distance(a.lower(), b.lower()) for a, b in zip(y_true, y_pred)]
    average_distance = sum(distances) / len(distances)
    print(f"Average Levenshtein Distance: {average_distance:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
```

- **Precision, Recall, F1-Score**: These metrics are calculated using `sklearn`.
- **Levenshtein Distance**: Measures the similarity between true and predicted names.

### Prediction

The model predicts the redacted names, incorporating a heuristic to filter potential candidates based on redaction length:

```python
def make_predictions(model, vectorizer, data):
    X_test = list(data['features'])
    X_vectorized = vectorizer.transform(X_test)
    class_labels = model.classes_

    name_lengths = {name: len(name.replace(' ', '').replace('.', '').replace("'", "").lower()) for name in class_labels}

    predictions = []
    for i, x in enumerate(X_test):
        redaction_length = x['redaction_length']
        candidate_names = [name for name in class_labels if abs(name_lengths[name] - redaction_length) <= 2]
        if not candidate_names:
            candidate_names = class_labels
        candidate_indices = [list(class_labels).index(name) for name in candidate_names]
        probs = model.predict_proba([X_vectorized[i]])[0]
        candidate_probs = probs[candidate_indices]
        best_candidate_index = candidate_indices[np.argmax(candidate_probs)]
        predicted_name = class_labels[best_candidate_index]
        predictions.append(predicted_name)

    results = pd.DataFrame({'id': data['id'], 'name': predictions})
    return results
```

- **Length Filtering**: Filters potential candidates based on redaction length to narrow down possible names.
- **Prediction Logic**: Chooses the name with the highest probability among the candidates.

---

## Usage Instructions

### Training and Evaluation

To train and evaluate the model, run the main script:

```sh
python unredactor.py
```

The script will train the model, evaluate it on the validation set, and save the trained model and vectorizer for later use.

### Generating Predictions

To generate predictions on the test dataset, ensure that the `test.tsv` file is in the root directory and execute the script. The predictions will be saved to `submission.tsv`.

---

## Examples

Here are some examples of how the model works, from feature extraction to prediction:

### Example 1: Feature Extraction

**Input Context**:
```
"I had a meeting with █████ yesterday to discuss the project timeline."
```

**Extracted Features**:
- **Previous Words**:
  - `prev_word_1`: 'with'
  - `prev_word_2`: 'meeting'
  - `prev_word_3`: 'a'
- **Next Words**:
  - `next_word_1`: 'yesterday'
  - `next_word_2`: 'to'
  - `next_word_3`: 'discuss'
- **POS Tags**:
  - `prev_pos_1`: 'IN'
  - `prev_pos_2`: 'NN'
  - `prev_pos_3`: 'DT'
  - `next_pos_1`: 'NN'
  - `next_pos_2`: 'TO'
  - `next_pos_3`: 'VB'
- **Redaction Length**: 5

### Example 2: Model Prediction

**Input Context**:
```
"I met with █████ yesterday to discuss our plans."
```

**Predicted Output**:
```
id    name
1     John Doe
```
The model predicts that the redacted name is likely "John Doe" based on the context and training data.

---

## Testing

### Running Tests

Tests are provided to evaluate each part of the code.

1. **Ensure pytest is Installed**

   ```
   pip install pytest
   ```

2. **Run the Tests**

   ```
   pytest tests/test_unredactor.py
   ```

3. **View Test Results**

   - Sample output:

     ```
     ============================= test session starts =============================
     collected 6 items

     tests/test_unredactor.py ......                                          [100%]

     ============================== 6 passed in 0.50s ==============================
     ```

### Test Functions

Here is a summary of the test functions included in `tests/test_unredactor.py`:

#### 1. `test_extract_features()`

```python
def test_extract_features():
    features = extract_features(sample_context)
    assert features is not None, "Features should not be None"
    assert features['prev_word_1'] == expected_features['prev_word_1'], "Incorrect previous word"
    assert features['next_word_1'] == expected_features['next_word_1'], "Incorrect next word"
    assert features['prev_pos_1'] == expected_features['prev_pos_1'], "Incorrect previous POS tag"
    assert features['next_pos_1'] == expected_features['next_pos_1'], "Incorrect next POS tag"
    assert features['redaction_length'] == expected_features['redaction_length'], "Incorrect redaction length"
```

This test verifies if the `extract_features()` function correctly extracts features from the context, such as previous and next words, part-of-speech tags, and redaction length.



#### 2. `test_load_and_preprocess_data()`

```python
def test_load_and_preprocess_data():
    data = pd.DataFrame({
        'split': ['training', 'validation'],
        'name': ['John Doe', 'Jane Smith'],
        'context': [sample_context, sample_context]
    })
    temp_file = 'temp_unredactor.tsv'
    data.to_csv(temp_file, sep='\t', index=False, header=False)

    loaded_data = load_and_preprocess_data(temp_file)
    os.remove(temp_file)

    assert not loaded_data.empty, "Loaded data should not be empty"
    assert 'features' in loaded_data.columns, "Features column should exist"
    assert len(loaded_data) == 2, "Loaded data should have 2 entries"
```

This test checks whether the `load_and_preprocess_data()` function successfully loads data, applies feature extraction, and handles preprocessing correctly.

#### 3. `test_train_model()`

```python
def test_train_model():
    X = [
        {'prev_word_1': 'with', 'next_word_1': 'yesterday', 'redaction_length': 5},
        {'prev_word_1': 'with', 'next_word_1': 'tomorrow', 'redaction_length': 7}
    ]
    y = ['John Doe', 'Jane Smith']

    model, vectorizer = train_model(X, y)
    assert isinstance(model, RandomForestClassifier), "Model should be a RandomForestClassifier"
    assert isinstance(vectorizer, DictVectorizer), "Vectorizer should be a DictVectorizer"
```

This test confirms that the `train_model()` function returns a trained `RandomForestClassifier` and a `DictVectorizer`.

#### 4. `test_evaluate_model()`

```python
def test_evaluate_model(capsys):
    X = [
        {'prev_word_1': 'with', 'next_word_1': 'yesterday', 'redaction_length': 5},
        {'prev_word_1': 'with', 'next_word_1': 'tomorrow', 'redaction_length': 7}
    ]
    y = ['John Doe', 'Jane Smith']
    model, vectorizer = train_model(X, y)

    evaluate_model(model, vectorizer, X, y)
    captured = capsys.readouterr()
    assert "Classification Report:" in captured.out, "Should print classification report"
```

This test validates the `evaluate_model()` function by checking if the classification report is printed to the console.

#### 5. `test_make_predictions()`

```python
def test_make_predictions():
    X_train = [
        {'prev_word_1': 'with', 'next_word_1': 'yesterday', 'redaction_length': 5},
        {'prev_word_1': 'with', 'next_word_1': 'tomorrow', 'redaction_length': 7}
    ]
    y_train = ['John Doe', 'Jane Smith']
    model, vectorizer = train_model(X_train, y_train)

    test_data = pd.DataFrame({
        'id': [1, 2],
        'features': X_train
    })

    results = make_predictions(model, vectorizer, test_data)
    assert 'id' in results.columns, "Results should contain 'id' column"
    assert 'name' in results.columns, "Results should contain 'name' column"
    assert len(results) == 2, "Results should have 2 entries"
```

This test verifies if the `make_predictions()` function can successfully predict names based on test data.

#### 6. `test_full_pipeline()`

```python
def test_full_pipeline(tmpdir):
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

    train_file = tmpdir.join('temp_unredactor.tsv')
    test_file = tmpdir.join('temp_test.tsv')
    train_data.to_csv(train_file, sep='\t', index=False, header=False)
    test_data.to_csv(test_file, sep='\t', index=False, header=False)

    training_data = load_and_preprocess_data(str(train_file))
    X_train = list(training_data['features'])
    y_train = training_data['name']

    model, vectorizer = train_model(X_train, y_train)
    test_data_loaded = load_and_preprocess_data(str(test_file), is_training=False)

    predictions = make_predictions(model, vectorizer, test_data_loaded)
    assert len(predictions) == 1, "Should have one prediction"
    assert predictions.iloc[0]['id'] == 1, "ID should be 1"
    assert predictions.iloc[0]['name'] in ['John Doe', 'Jane Smith'], "Name should be one of the trained names"
```

This test evaluates the entire pipeline, including data loading, preprocessing, training, and making predictions to ensure everything works seamlessly together.

---

## Evaluation Results

The evaluation of the model revealed the following metrics on the test data:

- **Precision**: 0.0413
- **Recall**: 0.0521
- **F1-Score**: 0.0418

These metrics indicate that the model is struggling with effectively identifying actor names from the input. The low precision implies that there were many false positives in the predictions, while the low recall points to the model missing many of the actual actor names present in the data.

The overall **accuracy** of the model was **0.05** (5%). The macro-average and weighted average metrics, both around 2-4%, similarly suggest that the model is not effectively capturing actor names across the dataset.

### Key Observations:

1. **Class Imbalance**: Many of the actors in the dataset had very few occurrences, leading to a severe class imbalance problem. The model had trouble generalizing across all the names, resulting in a lack of precision and recall, especially for actors with fewer samples.

2. **Data Complexity**: Some actor names had variations or were combined with additional text, leading to a higher rate of misclassification. Examples include different spellings of the same actor's name or actor names that were accompanied by additional descriptors (e.g., "Pierce Brosnan" vs. "Brosnan"). This inconsistency made it difficult for the model to identify names reliably.

3. **Low Recall**: A recall value of 0.0521 indicates that the model missed most of the actual actor names in the input data. This can be attributed to inadequate training data or insufficient complexity in the model architecture.

4. **Submission Results**: The results saved in `submission.tsv` included predictions for 200 samples from the test set. These predictions further highlighted the model's difficulty in distinguishing actor names accurately. Many predicted names were either incorrect or did not match the intended actor due to noise or lack of contextual understanding.

## Challenges

1. **Class Overlap and Similar Names**: Some actor names were highly similar, which led to incorrect predictions. For example, the model often confused "William Haines" with "William Powell" due to similarities in first names and the nature of their appearances in the dataset.

2. **Inconsistent Representation**: In the dataset, actor names sometimes appeared in different formats (e.g., full names, last names only, or with additional descriptors). This led to poor model generalization as it struggled to understand these variations.

3. **Model Complexity**: The model may have been too simplistic to adequately capture the complexity of actor names in a real-world context. A more advanced architecture, such as a transformer-based model, may help improve name identification by considering the context better.

## Future Improvements

To improve the model's performance and address the issues identified above, several steps can be taken:

1. **Increase Training Data**: Collecting more examples of actor names, especially for underrepresented names, would help balance the dataset and improve the model's ability to generalize.

2. **Data Preprocessing**: Standardizing the representation of actor names in the dataset (e.g., always using full names) could help the model learn more effectively. Techniques such as entity normalization could be applied to make all variations of a name consistent.

3. **Advanced Model Architecture**: Implementing a transformer-based model like BERT or GPT-2 could potentially improve the model's understanding of context and reduce misclassification. These models are better equipped to deal with the complexities of text and could significantly enhance recall and precision.

4. **Fine-Tuning Hyperparameters**: Fine-tuning the model's hyperparameters, such as learning rate and number of layers, might also help achieve better performance.

5. **Address Class Imbalance**: Techniques like oversampling or using synthetic data generation (e.g., SMOTE) could be used to handle the class imbalance issue.

## Conclusion

The current iteration of the model performed poorly in identifying actor names, achieving only around 5% accuracy. The low precision and recall values highlight the need for better data preprocessing, increased training data, and potentially more complex modeling approaches to improve performance. Future iterations will focus on enhancing data quality, employing more sophisticated model architectures, and addressing class imbalance to improve the overall effectiveness of the actor name classification task.

## Next Steps

- Investigate and apply more advanced text representation techniques, such as embeddings or transformer models.
- Address data inconsistencies through better preprocessing and normalization.
- Increase the dataset size and balance to ensure the model has a more even distribution of actor names to learn from.



## Evaluation Metrics

- **Precision**: Measures the accuracy of positive predictions.
- **Recall**: Measures the model's ability to find all positive instances.
- **F1-Score**: Harmonic mean of precision and recall, providing a balance between the two.
- **Average Levenshtein Distance**: Measures the average similarity between the true and predicted names, indicating how closely the predictions match the actual values.

---

## Collaborators
---
## Assumptions

The following assumptions were made during the development of the model and pipeline:

### Data Consistency

- **Redaction Pattern**: Redacted names are represented by block characters (`█`).
- **Data Format**:
  - **Training Data**: A TSV file (`unredactor.tsv`) with columns: `split`, `name`, and `context`.
    - `split`: Indicates whether the row is for training or validation (`training` or `validation`).
    - `name`: The original name that was redacted.
    - `context`: The text containing the redaction.
  - **Test Data**: A TSV file (`test.tsv`) with columns: `id` and `context`.
    - `id`: Unique identifier for each test instance.
    - `context`: The text containing the redaction.
- **Validation Set**: Identified within the training data by the `split` column labeled `validation`.
- The training and test data are assumed to be representative of the same distribution. The contexts and patterns in the training set are expected to be similar to those in the validation and test sets.
- Redacted names are replaced with a uniform block character (`█`), and this format remains consistent across the entire dataset.

### Feature Relevance

- Words immediately surrounding the redaction provide sufficient contextual cues for accurate prediction.
- Part-of-speech (POS) tags for the surrounding words are also assumed to be relevant for the prediction task.
- The length of the redaction (i.e., the number of block characters) is assumed to have a correlation with the length of the original name.

### Model Sufficiency

- The `RandomForestClassifier` is assumed to be sufficient for capturing the relationships between features and labels without extensive hyperparameter tuning.
- The chosen hyperparameters, such as the number of estimators and class weights, are assumed to provide good enough performance.

### Computational Resources

- Adequate computational resources are available to train the model on the entire dataset, which allows for using all available samples without reducing the dataset size.
- The runtime environment is stable, supporting all necessary dependencies as specified in the `Pipfile`.

### Operational Stability

- No interruptions will affect the training, evaluation, or prediction phases.
- Data loading and preprocessing functions are expected to handle input anomalies or corrupted data gracefully.

### Performance Metrics

- Metrics such as accuracy, precision, recall, and F1-score are appropriate for evaluating the model's performance.
- The evaluation metrics fairly represent the model's effectiveness across potentially imbalanced classes.
---
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: Ensure all paths and filenames correspond to your actual project setup. Adjust instructions accordingly if your directory structure or filenames differ.

