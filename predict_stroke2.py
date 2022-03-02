import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_model() : 
    with open("avc_complete_5.pkl.zip", "rb") as file :
        data = pickle.load(file)
    return data

data = load_model()

ran_for = data["model"]
le_Genre = data["le_Genre"]
le_Hypertension = data["le_Hypertension"]
le_maladie_cardiaque = data["le_maladie_cardiaque"]
le_status_fumeur = data["le_status_fumeur"]

st.title("MED x IA : Diagnostique rapide d'accident vasculaire cérébral")
st.write("""### Veuillez saisir des informations sur le patient : """)

Genre = (
    "Homme",
    "Femme",
    "Autre",
)
Genre = st.selectbox("Genre", Genre)

Âge = st.slider("Âge", 0, 100, 42)

Hypertension = ("Non", "Oui")
Hypertension = st.selectbox("Est-ce que le patient fait de l'hypertension?", Hypertension)

maladie_cardiaque = ("Non", "Oui")
maladie_cardiaque = st.selectbox("Est-ce que le patient a une maladie cardiaque?", maladie_cardiaque)

moy_taux_glucose = st.slider("Moyenne du taux de glucose", 40, 300, 105)

indice_masse_corporelle = st.slider("Indice de masse corporelle", 0, 80, 20)

status_fumeur = (
    "Je ne sais pas",
    "Jamais fumé",
    "Fumait",
    "Fume",
)
status_fumeur = st.selectbox("Statut de fumeur", status_fumeur)

ok = st.button("Calculer le risque d'AVC dans un futur proche")
if ok : 
    X = np.array([[Genre, Âge, Hypertension, maladie_cardiaque, moy_taux_glucose, indice_masse_corporelle, status_fumeur]])
    X[:,0] = le_Genre.transform(X[:,0])
    X[:,2] = le_Hypertension.transform(X[:,2])
    X[:,3] = le_maladie_cardiaque.transform(X[:,3])
    X[:, 6] = le_status_fumeur.transform(X[:,6])
    X = X.astype(float)

    avc = ran_for.predict_proba(X)
    st.text(f"Les risques d'AVC sont de {round(float(100*(avc[:,1])),2)} %.")
