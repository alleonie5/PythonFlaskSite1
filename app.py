import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request
from sklearn.cluster import KMeans, AgglomerativeClustering
import matplotlib
matplotlib.use("Agg") # Indispensable pour fonctionner sans interface graphique
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

app = Flask(__name__)

# --- ALGORITHME K-MEANS ---
def segmenter_kmeans(file, k):
    img = Image.open(file).convert("RGB")
    w, h = img.size
    scale = min(1.0, 400 / max(w, h))
    img_small = img.resize((int(w * scale), int(h * scale)))
    arr = np.array(img_small)
    pixels = arr.reshape(-1, 3)
    
    km = KMeans(n_clusters=k, n_init=5, random_state=42)
    labels = km.fit_predict(pixels)
    centers = km.cluster_centers_.astype(np.uint8)
    
    new_arr = centers[labels].reshape(arr.shape)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(img_small); ax1.set_title("Originale"); ax1.axis('off')
    ax2.imshow(new_arr); ax2.set_title(f"K-Means (K={k})"); ax2.axis('off')
    
    # Conversion en Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- ALGORITHME HIERARCHIQUE ---
def segmenter_hierarchique(file, k):
    img = Image.open(file).convert("RGB")
    img.thumbnail((150, 150)) 
    arr = np.array(img)
    pixels = arr.reshape(-1, 3)

    indices = np.random.choice(pixels.shape[0], min(2000, pixels.shape[0]), replace=False)
    sample = pixels[indices]
    Z = linkage(sample, method='ward')

    model = AgglomerativeClustering(n_clusters=k)
    labels = model.fit_predict(pixels)
    
    centers = np.array([pixels[labels == i].mean(axis=0) for i in range(k)], dtype=np.uint8)
    new_arr = centers[labels].reshape(arr.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(new_arr); ax1.set_title(f"Hierarchique (K={k})"); ax1.axis('off')
    dendrogram(Z, ax=ax2, truncate_mode='lastp', p=k, no_labels=True)
    ax2.set_title("Dendrogramme")

    # Conversion en Base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close()
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# --- ROUTES ---

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/map")
def map():
    return render_template("map.html")

@app.route("/repertoire")
def repertoire():
    return render_template("repertoire.html")

@app.route("/clusters", methods=["GET", "POST"])
def clusters():
    if request.method == "POST":
        k = int(request.form.get("k_value", 5))
        algo = request.form.get("algo")
        file = request.files.get("image_segmentation")
        
        if file and file.filename != "":
            # On traite le fichier directement en mémoire
            if algo == "hierarchique":
                data_img = segmenter_hierarchique(file, k)
            else:
                data_img = segmenter_kmeans(file, k)
                
            return render_template("clusters.html", graphique=data_img, k_actuel=k, algo_choisi=algo)
            
    return render_template("clusters.html", graphique=None)

if __name__ == "__main__":
    app.run(debug=True, port=5001)