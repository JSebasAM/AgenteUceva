import os

import joblib
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from app.feature_extractor import extract_features

etiquetas = ['rojo', 'azul']

# Cargar dataset
def load_data(dataset_path):
  y = []
  X = []

  for  etiqueta in etiquetas:
    directorio = os.path.join(dataset_path, etiqueta)

    for filename in os.listdir(directorio):
      full_path = os.path.join(directorio, filename)
      histograma = extract_features(full_path)

      X.append(histograma)
      y.append(etiqueta)

  return np.array(X),np.array(y)

def train_model(dataset_path, model_name):
  X, y = load_data(dataset_path)

  scaler = StandardScaler()
  scaler.fit(X)
  X = scaler.transform(X)

  X_train, X_test, y_train, y_test = train_test_split(
    X,y,test_size=0.33,random_state=1
  )

  lr = LogisticRegression()
  lr.fit(X_train,y_train)

  accuracy_test = lr.score(X_test,y_test)
  accuracy_train = lr.score(X_train,y_train)

  # reentrenar con todo el dataset
  lr.fit(X,y)

  os.makedirs("models", exist_ok=True)
  joblib.dump((lr,scaler), f"models/{model_name}.pkl")

  classes = list(set(y))

  return accuracy_train, accuracy_test, classes,

def classify_image(image_path, model_name):
  model, scaler = joblib.load(f"models/{model_name}.pkl")

  features = extract_features(image_path)
  features = scaler.transform(features.reshape(1,-1))

  prediction = model.predict(features)[0]
  confidence = model.predict_proba(features).max()

  return prediction, float(confidence)