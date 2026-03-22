import pickle
from feature_extractor import extract_features

model = pickle.load(open("phishing_model.pkl","rb"))

url = input("Enter URL: ")

features = extract_features(url)

prediction = model.predict([features])

if prediction[0] == 1:
    print("⚠️ This URL is likely PHISHING")
else:
    print("✅ This URL appears LEGITIMATE")