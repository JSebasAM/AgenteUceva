import os
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Extraer el color dominante
def extraer_hist (ruta_imagen):
  image = cv2.imread(ruta_imagen)

  hist = cv2.calcHist([image],[1,2],None,[256,256],[0,256,0,256])
  return cv2.merge([hist])

def procesar(train_labels):
  y = []
  X = []
  for  train_label in train_labels:
    directorio = 'datasets/train/'+train_label

    for filename in os.listdir(directorio):
      full_path = directorio+'/'+filename
      histograma = np.hstack(extraer_hist(full_path))
      X.append(histograma)
      y.append(train_label)
  return X,y

etiquetas = ['saludable','nosaludable']

X, y = procesar(etiquetas)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33,random_state=1)

lr = LogisticRegression()
print(lr.fit(X_train,y_train))

print("Accuracy test: ",lr.score(X_test,y_test))

print("Accuracy train:", lr.score(X_train,y_train))



imagen_test_ruta = 'datasets/test/saludable.jpg'
imagen_test =  np.hstack(extraer_hist (imagen_test_ruta))

caracteristicas = scaler.transform(imagen_test.reshape(1,-1))

print("Predicción: ",lr.predict(caracteristicas))