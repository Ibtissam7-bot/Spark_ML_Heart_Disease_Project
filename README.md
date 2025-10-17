# Prédiction des Maladies Cardiovasculaires avec PySpark ML

## Description du projet

Ce projet utilise PySpark ML pour créer un pipeline d'apprentissage automatique capable de prédire les risques de maladies cardiovasculaires à partir de données médicales de routine.

### Objectifs
- Préparer des données médicales
- Réaliser une analyse exploratoire des données
- Entraîner et comparer 2 modèles de classification
- Identifier les facteurs de risque les plus importants pour le modèle d'arbre
- Créer un modèle déployable pour prédictions futures (Pipeline complète)

## Technologies utilisées

- **PySpark 3.x** : Traitement distribué et ML
- **Python 3.8+**
- **Matplotlib/Seaborn** : Visualisations
- **Pandas/NumPy** : Manipulation de données
- **Google Colab** : Environnement d'exécution

## Dataset

### Variables d'entrée
- Age : age (years)
- Sex : gender (0 = Female, 1 = Male)
- Angina : chest pain (1 = Stable angina, 2 = Unstable angina, 3 = Other pains, 4 = Asymptomatic)
- Blood_Pressure : resting blood pressure (mmHg)
- Cholesterol : cholesterol levels (mg/dl)
- Glycemia : fasting blood sugar (0 = Less than 120 mg/dl, 1 = More than 120 mg/dl)
- ECG : electrocardiogram results (0 = Normal, 1 = Anomalies, 2 = Hypertrophy)
- Heart_Rate : maximum heart rate reached
- Angina_After_Sport : angina pectoris after physical exertion (0 = no, 1 = yes)
- ECG_Angina : measure of the angina pectoris on the electrocardiogram
- ECG_Slope : slope on the electrocardiogram (1 = Rising, 2 = Stable, 3 = Falling)
- Fluoroscopy : fluoroscopy results (0 = No anomaly, 1 = Low, 2 = Medium, 3 = High)
- Thalassemia : presence of a Thalassaemia (3 = No, 6 = Thalassaemia under control, 7 = Unstable
Thalassaemia)
- Disease : presence of a cardiovascular disease (0 = No, 1/2/3/4 = Yes)

### Variable cible
- **Disease** : Présence de maladie cardiovasculaire (0 = Non, 1,2,3,4 = Oui)

## Installation et exécution

### Vérification sue l'installation des bibliothèques
```bash
pip install pyspark pandas numpy matplotlib seaborn
```

### Dans Google Colab
1. Ouvrir le notebook 
2. Exécuter toutes les cellules
3. Les visualisations et résultats s'affichent automatiquement


## Structure du projet

```
cardiovascular-prediction/
│
├── README.md                          # Ce fichier
│
├── notebook/
│   ├──  Heart_diseases_prediction_Pyspark.ipynb # Notebook principal
│
├── data/
│   ├── heart-disease-68ec37d6b52cb588200595     # Dataset
│
├── Pipeline/
│   └── best_cardiovascular_model/    # Modèle sauvegardé (Pipeline complet)
│
├── visualisations/ # Representations graphiques des distributions, etc... .png 
 

```

## Méthodologie

### 1. Exploration des données (EDA)
- Statistiques descriptives
- Distribution des variables
- Détection d'outliers
- Équilibre des classes

### 2. Prétraitement
- Traitement des valeurs manquantes
- Encodage des variables catégorielles (StringIndexer + OneHotEncoder)
- Normalisation des variables numériques (StandardScaler)
- Feature engineering (création de ratios et produits)

### 3. Modélisation
Deux modèles ont été entraînés et comparés :

#### Logistic Regression
- Modèle linéaire de base
- Rapide et interprétable

#### Random Forest
- Ensemble de 100 arbres de décision
- Robuste aux outliers
- Fournit l'importance des features

### 4. Évaluation
Métriques calculées pour chaque modèle :
- **Accuracy** : Taux de bonnes prédictions
- **F1-Score** : Moyenne harmonique Precision/Recall
- **Precision** : Taux de vrais positifs parmi les positifs prédits
- **Recall** : Taux de vrais positifs détectés
- **Matrice de confusion** : Répartition détaillée des prédictions

## Résultats

### Comparaison des modèles

| Modèle | Accuracy | F1-Score | Precision | Recall |
|--------|----------|----------|-----------|--------|
| Logistic Regression |  0.873418 | 0.870964 | 0.879476 | 0.873418 |
| Random Forest | 0.835443 |0.834123    |0.835314 | 0.835443
 |
### Meilleur modèle
**Logistic regression** avec un F1-Score de **0.870964**

## Auteur

**SANNAKY Ibtissam**
- Email: bissamsannaky@gmail.com
