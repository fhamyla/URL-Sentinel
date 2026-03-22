import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

from feature_extractor import extract_features

data = pd.read_csv("dataset.csv")

urls = data["url"]
labels = data["label"]

labels = labels.map({"legitimate":0, "phishing":1})

X = []

for url in urls:
    features = extract_features(url)
    X.append(features)

y = labels

X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train,y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test,pred)

print("Model accuracy:",accuracy)

pickle.dump(model,open("phishing_model.pkl","wb"))

print("Model saved successfully.")