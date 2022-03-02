import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import itertools
import seaborn as sns
import warnings
import pickle

import streamlit as st
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score
from pylab import rcParams
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from matplotlib import pyplot as plt
%matplotlib inline

rcParams["figure.figsize"] = 10, 6
warnings.filterwarnings("ignore")
sns.set(style="darkgrid")

def generate_model_report(y_actual, y_predicted) :
    print("Accuracy = ", accuracy_score(y_actual, y_predicted))
    print("Precision = ", precision_score(y_actual, y_predicted))
    print("Recall = ", recall_score(y_actual, y_predicted))
    print("F1 Score = ", f1_score(y_actual, y_predicted))
    pass
  
def generate_auc_roc_curve(model, x_test): 
    y_pred_probability = model.predict_probability(X_test)[:,1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_probability)
    auc = roc_auc_score(y_test, y_pred_probability)
    plt.plot(fpr, tpr, label = "AUC ROC Curve with Area under the curve = "+str(auc))
    plt.show()
    pass
  
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

df.rename(columns={'gender': 'Genre', 'age':'Âge', 'hypertension':'Hypertension', 
                  'heart_disease':'maladie_cardiaque', 'ever_married':'déjà_marié', 
                  'work_type':'type_travail', 'Residence_type':'type_résidence',
                  'avg_glucose_level' : 'moy_taux_glucose',
                  'bmi':'indice_masse_corporelle', 'smoking_status':'status_fumeur', 
                  'stroke':'avc'}, inplace=True)
df.Genre[df.Genre=='Male']='Homme'
df.Genre[df.Genre=='Female']='Femme'
df.Genre[df.Genre=='Other']='Autre'
df.déjà_marié[df.déjà_marié=='Yes']='Oui'
df.déjà_marié[df.déjà_marié=='No']='Non'
df.type_travail[df.type_travail=='Private']='Privé'
df.type_travail[df.type_travail=='Govt_job']='Gouvernement'
df.type_travail[df.type_travail=='Self-employed']='Indépendant'
df.type_travail[df.type_travail=='children']='Enfant'
df.type_travail[df.type_travail=='Never_worked']='Jamais travaillé'
df.type_résidence[df.type_résidence=='Urban']='Urbain'
df.type_résidence[df.type_résidence=='Rural']='Rural'
df.status_fumeur[df.status_fumeur=='Unknown']='Je ne sais pas'
df.status_fumeur[df.status_fumeur=='never smoked']='Jamais fumé'
df.status_fumeur[df.status_fumeur=='formerly smoked']='Fumais'
df.status_fumeur[df.status_fumeur=='smokes']='Fume'
df.Hypertension[df.Hypertension==1]='Oui'
df.Hypertension[df.Hypertension==0]='Non'
df.maladie_cardiaque[df.maladie_cardiaque==1]='Oui'
df.maladie_cardiaque[df.maladie_cardiaque==0]='Non'

df.drop(["id"], axis=1, inplace = True)
df.drop(["type_travail", "type_résidence", "déjà_marié"], axis=1, inplace = True)
df = df[df["indice_masse_corporelle"].notnull()]
df = df[df["indice_masse_corporelle"] <= 70]

le_Genre = LabelEncoder()
df["Genre"] = le_Genre.fit_transform(df["Genre"])
df["Genre"].unique()
le_Hypertension = LabelEncoder()
df["Hypertension"] = le_Hypertension.fit_transform(df["Hypertension"])
df["Hypertension"].unique()
le_maladie_cardiaque = LabelEncoder()
df["maladie_cardiaque"] = le_maladie_cardiaque.fit_transform(df["maladie_cardiaque"])
df["maladie_cardiaque"].unique()
le_status_fumeur = LabelEncoder()
df["status_fumeur"] = le_status_fumeur.fit_transform(df["status_fumeur"])
df["status_fumeur"].unique()

target = "avc"
X = df.drop(["avc"], axis = 1)
y = df["avc"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

minority_class_len = len(df[df[target] == 1])
print(minority_class_len)
majority_class_indices = df[df[target] == 0].index
print(majority_class_indices)
random_majority_indices = np.random.choice(majority_class_indices,
                                          minority_class_len,
                                          replace=False)
minority_class_indices = df[df[target] == 1].index
print(minority_class_indices)
under_sample_indices = np.concatenate([minority_class_indices,
                                      random_majority_indices])
under_sample = df.loc[under_sample_indices]
X_train = under_sample.loc[:, df.columns!=target]
y_train = under_sample.loc[:, df.columns==target]

ran_for = RandomForestClassifier(n_estimators=2500, random_state = 200)
ran_for.fit(X_train, y_train)

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
