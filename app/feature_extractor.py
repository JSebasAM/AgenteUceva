import cv2
import numpy as np

def extract_features(img_path):

    image = cv2.imread(img_path)

    if image is None:
        raise ValueError(f"No se pudo cargar la imagen: {img_path}")

    # Redimensionar
    image = cv2.resize(image, (128, 128))

    #Histograma por canal (RGB)

    hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
    hist_b = cv2.calcHist([image], [0], None, [32], [0, 256])

    #Normalizamos los datos

    hist_r = cv2.normalize(hist_b, hist_b).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()

    #Concatenamos los datos

    return np.hstack(cv2.merge((hist_r, hist_g, hist_b)))

