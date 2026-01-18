import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data/Iris.csv")

# Drop Id column
df = df.drop(columns=["Id"])

# Features and target
X = df.drop("Species", axis=1)
y = df["Species"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline (scaler + model)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(random_state=42))
])

# Train model
pipeline.fit(X_train, y_train)

# Save model
joblib.dump(pipeline, "model/iris_model.pkl")

print("✅ Model trained and saved successfully")
