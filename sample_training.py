from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib

texts = [
    "Aadhar Number: 1234 5678 9012", 
    "Permanent Account Number", 
    "Account Holder Name Central Bank", 
    "Family Income Certificate", 
    "High School Examination", 
    "This is to certify that the student has paid the fee",
    "Disability 40% certificate",
    "Certified that Mr. has died"
]

labels = [
    "Aadhar Card", 
    "PAN Card", 
    "Bank Passbook", 
    "Income Certificate", 
    "Marksheet", 
    "Fee Receipt", 
    "Disability Certificate",
    "Death Certificate"
]

model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, labels)

joblib.dump(model, "model/classifier.pkl")
