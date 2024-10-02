import time
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import tree

# Set the page configuration
st.set_page_config(page_title="Trafic Analyse", page_icon="🔍")

# Fonction pour charger et afficher un extrait du fichier CSV
def load_data(file):
    data = pd.read_csv(file)
    return data

# Fonction pour préparer les données
def prepare_data(train, test):
    # Suppression de la colonne redondante
    train.drop(['num_outbound_cmds'], axis=1, inplace=True)
    test.drop(['num_outbound_cmds'], axis=1, inplace=True)

    # Standardisation des données numériques
    scaler = StandardScaler()
    cols = train.select_dtypes(include=['float64', 'int64']).columns
    sc_train = scaler.fit_transform(train.select_dtypes(include=['float64', 'int64']))
    sc_test = scaler.transform(test.select_dtypes(include=['float64', 'int64']))
    
    sc_traindf = pd.DataFrame(sc_train, columns=cols)
    sc_testdf = pd.DataFrame(sc_test, columns=cols)

    # Encodage des variables catégorielles
    encoder = LabelEncoder()
    cattrain = train.select_dtypes(include=['object']).copy()
    cattest = test.select_dtypes(include=['object']).copy()
    traincat = cattrain.apply(encoder.fit_transform)
    testcat = cattest.apply(encoder.fit_transform)

    # Séparation des colonnes cibles
    enctrain = traincat.drop(['class'], axis=1)
    train_x = pd.concat([sc_traindf, enctrain], axis=1)
    train_y = train['class']

    test_df = pd.concat([sc_testdf, testcat], axis=1)

    return train_x, train_y, test_df

# Fonction d'analyse du trafic réseau
def analyze_traffic(train_x, train_y, test_df):
    # Initialisation des modèles
    knn_classifier = KNeighborsClassifier(n_jobs=-1)
    lgr_classifier = LogisticRegression(n_jobs=-1, random_state=0, max_iter=1000)
    bnb_classifier = BernoulliNB()
    dtc_classifier = tree.DecisionTreeClassifier(criterion='entropy', random_state=0)

    # Entraînement des modèles
    knn_classifier.fit(train_x, train_y)
    lgr_classifier.fit(train_x, train_y)
    bnb_classifier.fit(train_x, train_y)
    dtc_classifier.fit(train_x, train_y)

    # Prédiction avec le KNeighborsClassifier
    pred_knn = knn_classifier.predict(test_df)
    
    return pred_knn

# Fonction pour afficher un graphique natif avec Streamlit
def plot_result_summary_streamlit(predictions):
    # Compter les valeurs uniques des prédictions
    result_summary = pd.Series(predictions).value_counts()

    # Afficher un graphique avec st.bar_chart
    st.bar_chart(result_summary)

# Fonction pour afficher la barre de progression
def display_progress(predictions):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(predictions)
    correct_preds = 0

    for i in range(total):
        progress = (i + 1) / total
        progress_bar.progress(progress)
        status_text.text(f"Analyse en cours... {int(progress * 100)}%")
        
        # Analyse des prédictions sans pause
        if predictions[i] == 'normal':  # Remplacez 'normal' par la condition correcte
            correct_preds += 1

    accuracy = (correct_preds / total) * 100
    status_text.text(f"Analyse terminée avec une précision de {accuracy:.2f}%")
    st.success(f"Analyse terminée avec une précision de {accuracy:.2f}%")
    st.write(f"Précision finale : {accuracy:.2f}%")
    st.progress(accuracy / 100)


# Interface Streamlit
st.title("Analyse du Trafic Réseau")

# Chargement du fichier d'entraînement (Train_data.csv déjà présent)
train_data = pd.read_csv("Train_data.csv")

# Sélection du fichier test par l'utilisateur
test_file = st.file_uploader("Télécharger le fichier test_data.csv", type="csv")

if test_file is not None:
    # Charger et afficher un extrait du fichier test
    test_data = load_data(test_file)
    st.write("Extrait des données du fichier :")
    st.write(test_data.head(4))

    # Bouton pour lancer l'analyse
    if st.button("Analyser le trafic"):
        with st.spinner("Préparation des données..."):
            train_x, train_y, test_df = prepare_data(train_data, test_data)
        
        with st.spinner("Analyse en cours..."):
            results = analyze_traffic(train_x, train_y, test_df)

        # Affichage des résultats des 17 premières lignes
        # Créer une liste pour les résultats
        results_list = []
        for i, result in enumerate(results[:17]):
            if result == 'anomaly':
                results_list.append({"Intrusion": "Détectée (Alerte)", "Ligne": i + 1})
            else:
                results_list.append({"Intrusion": "Normal (Rien à signaler)", "Ligne": i + 1})

        # Convertir la liste en DataFrame
        results_df = pd.DataFrame(results_list)

        # Afficher les résultats dans un tableau
        st.write("Résultats de l'analyse pour les premières lignes :")
        st.dataframe(results_df)  # ou st.table(results_df) pour un affichage statique

        # Afficher la barre de progression et la précision
        display_progress(results)

        # Afficher le graphique résumé
        st.write("Résumé global des résultats :")
        plot_result_summary_streamlit(results)