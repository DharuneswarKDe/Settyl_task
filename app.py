#importing modules
from fastapi import FastAPI
from pydantic import BaseModel
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware


from sklearn.feature_extraction.text import TfidfVectorizer
import containerdata
internal_values=containerdata.internal

nltk.download('stopwords')
nltk.download('wordnet')

origins = [
    "http://127.0.0.1:3000/k:/New folder (2)/formpage.html",
    "http://localhost:8080",  
]



#Declaring FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

#input/output models using Pydantic
class request_body(BaseModel):
    inp_text:str

class PredictionResponse(BaseModel):
    class_name: str

#handling url 
@app.get('/')
def main():
    return {'message': 'Container data prediction!'}
model_rf=joblib.load("model_rf.joblib")

@app.post('/predict',response_model=PredictionResponse)
def predict(data : request_body):
    input_text=data.inp_text
    processed_text=preprocess_input(input_text)
    vectorizer = TfidfVectorizer(max_features=1000)
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    input_tfidf = vectorizer.transform([processed_text])
    prediction = model_rf.predict(input_tfidf)
    class_text=str(internal_values[prediction])
    return JSONResponse(content={"class_name": class_text})

#preprocessing input_text
def preprocess_input(text):
   #normalising data
    text =text.lower()
    #keeping only alphabets
    text=re.sub(r'[^a-zA-Z]', ' ', text)
    #tokenization
    text = ' '.join(text.split())  # Split and rejoin
    #removed stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    #Stemmer and Lemmatizer
    snowball_stemmer = SnowballStemmer("english")
    wordnet_lemmatizer = WordNetLemmatizer()
    stemmed_text = ' '.join(snowball_stemmer.stem(word) for word in text.split())
    lemmatized_text = ' '.join(wordnet_lemmatizer.lemmatize(word) for word in stemmed_text.split())
    return lemmatized_text