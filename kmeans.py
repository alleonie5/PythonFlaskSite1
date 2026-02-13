import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# 1. Configuration du chemin
dossier = "C:\\Users\\leoni\\OneDrive\\Документы\\DEUST WMI 2025-2026\\cours photoshop\\Double exposition\\" # Remplacez par votre nom de dossier
fichier = "Ca sent le sapin.jpeg" # Remplacez par votre nom de fichier
chemin_complet = os.path.join(dossier, fichier)

# 2. Chargement et conversion
try:
    img = Image.open(chemin_complet)
    img_np = np.array(img)
except FileNotFoundError:
    print(f"Erreur : Le fichier {chemin_complet} n'a pas été trouvé.")
    exit()

# 3. Préparation des données pour le ML
# On transforme l'image (Hauteur, Largeur, RGB) en une longue liste de pixels
h, w, c = img_np.shape
pixels = img_np.reshape(-1, c)

# 4. Application du K-Means
# n_clusters représente le nombre de couleurs finales (segments)
k = 2 
model = KMeans(n_clusters=k, n_init=10)
labels = model.fit_predict(pixels)

# On remplace chaque pixel par la couleur moyenne de son groupe
centres_couleurs = model.cluster_centers_.astype(np.uint8)
image_segmentee = centres_couleurs[labels].reshape(h, w, c)

# 5. Affichage
plt.figure(figsize=(15, 10))
plt.subplot(1, 2, 1)
plt.title("Image Originale")
plt.imshow(img_np)

plt.subplot(1, 2, 2)
plt.title(f"Segmentation (K={k})")
plt.imshow(image_segmentee)
plt.show()