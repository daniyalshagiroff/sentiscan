from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from data_load import load_data
from preprocessing import clean_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

def train_model(X, y):
    param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'penalty': ['l2'],
    'solver': ['liblinear', 'saga'],
    'max_iter': [100, 200],
    'class_weight': [None, 'balanced']
    }
    regression = LogisticRegression()
    grid_search = GridSearchCV(regression, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    print("Grid Search completed. Results:")
    print("Best params:", grid_search.best_params_)
    print("Best accuracy:", grid_search.best_score_)
    print("\n")
    print("\n")
    best_model = grid_search.best_estimator_
    best_model.fit(X, y)
    return best_model

def train_and_evaluate_model(data_path, model_path, vectorizer_path):
    print("train_and_evaluate_model called")
    # 1. Load the data
    df = load_data(data_path)

    # 2. Clean the text and remove NaN values
    df["cleaned"] = df[3].apply(clean_text)
    df = df.dropna(subset=["cleaned"])

    # Separate features (X) and target labels (y)
    X = df["cleaned"]
    y = df[2]

    print("Cleaning completed.")

    # 3. Vectorize the text data
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)
    print("Vectorization completed.")

    # 4. Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_vec, y, test_size=0.2, random_state=42)
    print("Splitting completed.")

    # 5. Train the model
    model = train_model(X_train, y_train)
    print("Training completed!")

    # 6. Save the model and vectorizer
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print("Model and vectorizer saved!")

    # 7. Make predictions on the validation set
    y_pred = model.predict(X_val)

    # 8. Calculate accuracy
    accuracy = accuracy_score(y_val, y_pred)
    print("Accuracy:", accuracy)


# Example usage
train_and_evaluate_model(
    data_path="data/raw/twitter_training.csv",
    model_path="models/logreg.pkl",
    vectorizer_path="models/vectorizer.pkl"
)