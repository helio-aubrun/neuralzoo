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