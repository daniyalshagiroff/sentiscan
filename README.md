 # SentiScan

SentiScan is a machine learning project designed to train and evaluate a sentiment analysis model using logistic regression. The project includes data preprocessing, model training, and validation on a separate dataset.

## Features
- **Data Preprocessing**: Text cleaning, lemmatization, and removal of stopwords.
- **Model Training**: Logistic regression with hyperparameter tuning using GridSearchCV.
- **Validation**: Evaluate the trained model on a separate dataset.
- **Reusable Components**: Modular code for data loading, preprocessing, training, and validation.

## Project Structure
sentiscan/ ├── data/ │ ├── raw/ # Raw datasets │ ├── processed/ # Processed datasets ├── src/ │ ├── data_load.py # Data loading utilities │ ├── preprocessing.py # Text preprocessing functions │ ├── training.py # Model training and evaluation │ ├── validation.py # Model validation on new datasets ├── models/ # Saved models and vectorizers


## Requirements
- **Python**: 3.8+
- **Required Libraries**:
  - `pandas`
  - `scikit-learn`
  - `nltk`
  - `joblib`

## Usage
1. Train the Model
Run the src/training.py script to train the model:
- python training.py
This will:
Load the training dataset (data/raw/twitter_training.csv).
Preprocess the text data.
Train a logistic regression model.
Save the trained model and vectorizer to the models directory.

## Key Functions
load_data(path): Loads a dataset from the specified path.
clean_text(text): Cleans and preprocesses text data.
train_model(X, y): Trains a logistic regression model with hyperparameter tuning.
evaluate_model_on_new_data(model_path, vectorizer_path, test_data_path): Evaluates the model on a new dataset.
Example Output

## Example Output
Training:
Cleaning completed.
Vectorization completed.
Splitting completed.
Training completed!
Model and vectorizer saved!
Accuracy: 0.85

Validation:
Model and vectorizer loaded.
Test data loaded.
Text cleaning completed.
Vectorization of test data completed.
Accuracy on new dataset: 0.83
