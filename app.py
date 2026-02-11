import os  # Ajouté : pour lire tes dossiers
from flask import Flask, render_template, request  # Ajouté : request pour la liste déroulante

app = Flask(__name__)

# Le chemin vers tes images
CHEMIN_IMAGES = 'static/uploads'

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/map")
def afficher_ma_carte():
    return render_template("map.html")

@app.route("/repertoire")
def repertoire():
    return render_template("repertoire.html")

if __name__ == '__main__':
    app.run(debug=True, port=5001)