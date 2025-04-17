from sklearn.metrics import accuracy_score
import joblib
from data_load import load_data
from preprocessing import clean_text

# Проверка модели на другом датасете
def evaluate_model_on_new_data(model_path, vectorizer_path, test_data_path):
    # Загрузка модели и векторизатора
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    print("Model and vectorizer loaded.")

    # Загрузка нового датасета
    test_df = load_data(test_data_path)
    print("Test data loaded.")

    # Очистка текста
    test_df["cleaned"] = test_df[3].apply(clean_text)
    test_df = test_df.dropna(subset=["cleaned"])
    X_test = test_df["cleaned"]
    y_test = test_df[2]
    print("Text cleaning completed.")

    # Преобразование текста в векторное представление
    X_test_vec = vectorizer.transform(X_test)
    print("Vectorization of test data completed.")

    # Предсказания
    y_pred = model.predict(X_test_vec)

    # Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy on new dataset:", accuracy)

# Load the saved model and vectorizer, and evaluate them on a new dataset
evaluate_model_on_new_data(
    model_path="models/logreg.pkl",
    vectorizer_path="models/vectorizer.pkl",
    test_data_path="data/raw/twitter_validation.csv"
)