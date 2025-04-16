from data_load import load_data
from preprocessing import clean_text
from training import train_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# 1. Load the data
df = load_data("data/raw/twitter_training.csv")

# 2. Clean the text and remove NaN values
df["cleaned"] = df[3].apply(clean_text)
df = df.dropna(subset=["cleaned"])
print(df["cleaned"].head())
X = df["cleaned"]
y = df[2]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_vec, y, test_size=0.2, random_state=42)
# 3. Vectorize the text data

# 4. Train the model
model = train_model(X_train, y_train)

# 5. Save the model and vectorizer
import joblib
joblib.dump(model, "models/logreg.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

# Make predictions on the test set
y_pred = model.predict(X_val)

# Calculate accuracy
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)