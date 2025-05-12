import joblib

model = joblib.load("model/classifier.pkl")

def predict_document_type(text: str) -> str:
    return model.predict([text])[0]
