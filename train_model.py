import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("Data_final.csv")

encoder = LabelEncoder()
data["Career"] = encoder.fit_transform(data["Career"])

X = data.drop("Career", axis=1)
y = data["Career"]

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")

print("✅ Model Saved")
