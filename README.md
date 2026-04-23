# Perceptron Multicouches (MLP)

## Qu'est-ce qu'un MLP ?

Le **Perceptron Multicouches** (Multilayer Perceptron, MLP) est le modèle fondateur des réseaux de neurones artificiels. Il appartient à la famille des réseaux de neurones feedforward : l'information y circule dans un seul sens, de la couche d'entrée vers la couche de sortie, sans boucle de rétroaction.

---

## Architecture

Un MLP est composé de plusieurs **couches de neurones** organisées en séquence.

```
[Entrée] ──► [Couche cachée 1] ──► [Couche cachée 2] ──► ... ──► [Sortie]
```

### Couche d'entrée

- Reçoit les données brutes (pixels, mesures, mots encodés…)
- Ne réalise **aucun calcul** : elle transmet simplement les valeurs
- Le nombre de neurones est fixé par la **dimension des données** (ex. : 784 neurones pour une image 28×28)

### Couches cachées (denses / fully connected)

- Constituent le cœur du réseau, là où l'apprentissage a lieu
- Chaque neurone est connecté à **tous** les neurones de la couche précédente → couche dite **dense**
- Chaque connexion porte un **poids** (*w*) ; chaque neurone possède un **biais** (*b*)
- Le neurone calcule une somme pondérée, puis applique une **fonction d'activation** :

```
sortie = f( w₁x₁ + w₂x₂ + ... + wₙxₙ + b )
```

- La fonction d'activation introduit la **non-linéarité** indispensable pour apprendre des relations complexes
- Un MLP peut avoir une ou plusieurs couches cachées — au-delà de deux, on parle de réseau **profond** (deep learning)

Fonctions d'activation courantes :

| Fonction   | Formule                         | Usage typique              |
|------------|---------------------------------|----------------------------|
| ReLU       | `max(0, x)`                     | Couches cachées (défaut)   |
| Sigmoid    | `1 / (1 + e⁻ˣ)`                | Sortie binaire             |
| Tanh       | `(eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`     | Couches cachées (NLP…)     |
| Softmax    | `eˣⁱ / Σeˣʲ`                   | Sortie multiclasse         |
| LeakyReLU  | `max(αx, x)`                    | Variante robuste de ReLU   |

### Couche de sortie

- Produit la **prédiction finale**
- Son architecture dépend de la tâche :

| Tâche                        | Neurones de sortie | Activation |
|------------------------------|--------------------|------------|
| Régression                   | 1                  | Linéaire   |
| Classification binaire       | 1                  | Sigmoid    |
| Classification multiclasse   | N (nb de classes)  | Softmax    |

---

## Hyperparamètres

Les hyperparamètres sont fixés **avant l'entraînement** (contrairement aux poids, qui sont appris par rétropropagation).

### Architecture du réseau

| Hyperparamètre              | Description                                      |
|-----------------------------|--------------------------------------------------|
| Nombre de couches cachées   | Profondeur du réseau                             |
| Nombre de neurones/couche   | Largeur de chaque couche                         |
| Fonction d'activation       | Non-linéarité appliquée après chaque couche      |

### Entraînement

| Hyperparamètre         | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| Taux d'apprentissage   | Taille des pas lors de la descente de gradient *(paramètre le plus critique)* |
| Nombre d'epochs        | Nombre de passages complets sur le dataset                                  |
| Taille du batch        | Nombre d'exemples traités avant chaque mise à jour des poids                |
| Optimiseur             | Algorithme de mise à jour des poids (SGD, Adam, RMSProp…)                   |

### Régularisation (prévention du surapprentissage)

| Hyperparamètre         | Description                                                    |
|------------------------|----------------------------------------------------------------|
| Dropout                | Taux de neurones désactivés aléatoirement à chaque passe       |
| L1 / L2                | Pénalise les grands poids pour simplifier le modèle            |
| Batch normalization    | Normalise les activations entre les couches                    |

### Initialisation

| Hyperparamètre             | Description                                              |
|----------------------------|----------------------------------------------------------|
| Méthode d'initialisation   | Stratégie de départ des poids (Xavier/Glorot, He…)       |

---

## Trouver les bons hyperparamètres

En pratique, la recherche des hyperparamètres optimaux est souvent l'étape la plus coûteuse. Les approches courantes sont :

- **Grid search** : teste toutes les combinaisons d'une grille prédéfinie
- **Random search** : échantillonne aléatoirement dans l'espace des hyperparamètres
- **Optimisation bayésienne** : guide la recherche intelligemment selon les résultats précédents (ex. Optuna, Ray Tune) 

## Choix de l'architecture selon la tâche

L'architecture d'un MLP n'est pas universelle — elle s'adapte à la nature du problème. La couche de sortie change selon qu'on cherche à prédire une valeur continue (régression) ou à assigner une classe (classification). Les couches cachées, elles, suivent des règles empiriques communes.

---

### Régression

L'objectif est de prédire une ou plusieurs valeurs continues (prix, température, salaire…).

- **Neurones de sortie** : 1 (ou N si plusieurs valeurs à prédire simultanément)
- **Activation de sortie** : linéaire (aucune contrainte sur la valeur produite)
- **Fonction de perte** : MSE (Mean Squared Error) ou MAE (Mean Absolute Error)

> Exemple : prédire le prix d'un logement, estimer une température.

---

### Classification binaire

L'objectif est de séparer deux classes (spam/non-spam, malade/sain…).

- **Neurones de sortie** : 1
- **Activation de sortie** : sigmoid — écrase la valeur dans ]0, 1[, interprétable comme une probabilité
- **Seuil de décision** : 0.5 par défaut
- **Fonction de perte** : entropie croisée binaire

> Exemple : détection de spam, diagnostic médical binaire.

---

### Classification multiclasse

L'objectif est d'attribuer l'une de N classes possibles (chiffre de 0 à 9, espèce animale, langue…).

- **Neurones de sortie** : N (un par classe)
- **Activation de sortie** : softmax — normalise les sorties en une distribution de probabilités sommant à 1
- **Classe prédite** : celle avec la probabilité la plus élevée
- **Fonction de perte** : entropie croisée catégorielle

> Exemple : reconnaissance de chiffres manuscrits, classification d'images.

---

### Tableau récapitulatif

| Tâche                      | Neurones de sortie | Activation de sortie | Fonction de perte              |
|----------------------------|--------------------|----------------------|--------------------------------|
| Régression                 | 1 (ou N valeurs)   | Linéaire             | MSE / MAE                      |
| Classification binaire     | 1                  | Sigmoid              | Entropie croisée binaire       |
| Classification multiclasse | N classes          | Softmax              | Entropie croisée catégorielle  |

---

### Règles empiriques pour les couches cachées

Ces règles s'appliquent indépendamment de la tâche :

- **Largeur** : souvent comprise entre la taille de la couche d'entrée et celle de la sortie. Un réseau en entonnoir (qui rétrécit progressivement) est une heuristique courante pour la classification.
- **Profondeur** : 1 couche suffit pour des problèmes simples ; 2 à 3 couches couvrent la grande majorité des cas pratiques ; davantage est réservé aux données très structurées (images, texte).
- **Activation des couches cachées** : ReLU (`max(0, x)`) est le choix par défaut pour sa simplicité et sa robustesse face à la disparition du gradient. 