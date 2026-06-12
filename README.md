# Classification de Sentiments dans le Texte avec Naïve Bayes

Ce projet implémente un modèle de traitement automatique du langage naturel (NLP) pour classifier des critiques de textes (Amazon et IMDb) en sentiments **Positifs** ou **Négatifs**. L'approche repose sur l'algorithme probabiliste **Classification Naïve bayésienne (Multinomial Naive Bayes)**.

---

## 📌 Rappel Théorique (Principe Algorithmique)

L'algorithme s'appuie sur le théorème de Bayes et l'hypothèse "naïve" que tous les mots (tokens) d'un texte sont indépendants les uns des autres pour une classe donnée ($P(A|B) = P(A)$ si $A$ et $B$ sont indépendants).

Le pipeline de calcul suit ces 4 étapes clés :
1. **Étape 1 :** Calcul des probabilités a priori des classes, notées $P(Y)$ pour $Y \in \{0, 1\}$.
2. **Étape 2 :** Établissement du vocabulaire global sous forme de fréquences `["mot", count]`.
3. **Étape 3 :** Calcul de la vraisemblance des mots sachant la classe, soit la probabilité conditionnelle $P(\text{"mot"}|Y)$.
4. **Étape 4 :** Prédiction de la classe finale via la probabilité a posteriori : 
   $$P(Y|\text{"texte"}) \propto P(Y) \times P(\text{"texte"}|Y)$$

---

## 🛠️ Structure du Projet et Données

Le modèle charge deux jeux de données textuels séparés par des tabulations (`\t`) :
* **Entraînement (`train_df`) :** `amazon_cells_labelled.txt` (1 000 exemples équilibrés à 50% Positif / 50% Négatif).
* **Test (`test_df`) :** `imdb_labelled.txt` (748 exemples).

### Structure des fichiers attendue :
```text
├── votre_notebook.ipynb
└── data/
    └── sentiment+labelled+sentences/
        └── sentiment labelled sentences/
            ├── amazon_cells_labelled.txt
            └── imdb_labelled.txt
