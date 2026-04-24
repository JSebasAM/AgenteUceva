import os
import shutil
import zipfile
from io import BufferedReader, BufferedWriter

from fastapi import FastAPI, UploadFile, Form, File

from app.model import train_model, classify_image

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train")
def train(file:UploadFile = File(...), classifier_name: str = Form(...)):

    #Crea carpeta tmp para guardar temporalmente el archivo zip
    os.makedirs("tmp", exist_ok=True)

    tmp_path = f"tmp/{file.filename}"

    #Guarda temporalmente
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    extract_path = f"tmp/{classifier_name}"
    os.makedirs(extract_path, exist_ok=True)

    #Descomprimir
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    #Entrenar modelo
    accuracy, classes = train_model(extract_path, classifier_name)

    #remover archivos temporales zip, dataset
    os.remove(tmp_path)
    shutil.rmtree(extract_path)

    return {
        "classifier_name: ": classifier_name,
        "accuracy":  accuracy,
        "etiquetas reconocidas: ": classes
    }

@app.post("/classify")
def classify(file:UploadFile = File(...), classifier_name: str = Form(...)):

    os.makedirs("tmp", exist_ok=True)

    image_path = f"tmp/{file.filename}"
    with open(image_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction, confidence = classify_image(image_path, classifier_name)

    os.remove(image_path)

    return {
        "prediction: ": prediction,
        "confidence": confidence
    }