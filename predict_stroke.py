import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_model() : 
    with open("avc_complete_4.pkl", "rb") as file :
        data = pickle.load(file)
    return data

data = load_model()

ran_for = data["model"]
le_Genre = data["le_Genre"]
le_Hypertension = data["le_Hypertension"]
le_maladie_cardiaque = data["le_maladie_cardiaque"]
# le_déjà_marié = data["le_déjà_marié"] 
# le_type_travail = data["le_type_travail"]
# le_type_résidence = data["le_type_résidence"]
le_status_fumeur = data["le_status_fumeur"]

st.title("Soph-IA : Diagnostique rapide d'accident vasculo-cérébral")
st.write("""### Veuillez ajouter des informations sur le patient : """)

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

# déjà_marié = ("Non", "Oui")
# déjà_marié = st.selectbox("Est-ce que le patient a déjà été marié?", déjà_marié)

# type_travail = (
#     "Privé",
#     "Gouvernement",
#     "Indépendant",
#     "Enfant",
#     "Jamais travaillé",
# )
# type_travail = st.selectbox("Type de travail", type_travail)

# type_résidence = ("Urbain", "Rural")
# type_résidence = st.selectbox("Type de résidence", type_résidence)

moy_taux_glucose = st.slider("Moyenne du taux de glucose", 40, 300, 105)

indice_masse_corporelle = st.slider("Indice de masse corporelle", 0, 80, 30)

status_fumeur = (
    "Je ne sais pas",
    "Jamais fumé",
    "Fumais",
    "Fume",
)
status_fumeur = st.selectbox("Status de fumeur", status_fumeur)

ok = st.button("Calcule la possibilité d'avoir un avc dans un futur proche")
if ok : 
    X = np.array([[Genre, Âge, Hypertension, maladie_cardiaque, moy_taux_glucose, indice_masse_corporelle, status_fumeur]])
    X[:, 0] = le_Genre.transform(X[:,0])
    X[:,2] = le_Hypertension.transform(X[:,2])
    X[:,3] = le_maladie_cardiaque.transform(X[:,3])
    # X[:, 4] = le_déjà_marié.transform(X[:,4])
    # X[:, 5] = le_type_travail.transform(X[:,5])
    # X[:, 6] = le_type_résidence.transform(X[:,6])
    X[:, 6] = le_status_fumeur.transform(X[:,6])
    X = X.astype(float)

    avc = ran_for.predict_proba(X)
    st.text(f"Les chances d'avoir un avc sont de {round(float(100*(avc[:,1])),2)} %.")

echo "# exposcience1" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/sopleger/exposcience1.git
git push -u origin main