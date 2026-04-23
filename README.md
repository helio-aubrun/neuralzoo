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

## Glossaire — termes clés du MLP

---

### Fonction d'activation

Une fonction d'activation est appliquée à la sortie de chaque neurone pour introduire de la **non-linéarité** dans le réseau. Sans elle, empiler plusieurs couches reviendrait à une simple transformation linéaire, incapable d'apprendre des relations complexes.

| Fonction   | Formule                              | Plage de sortie | Usage typique                  |
|------------|--------------------------------------|-----------------|--------------------------------|
| ReLU       | `max(0, x)`                          | [0, +∞[         | Couches cachées (défaut)       |
| Sigmoid    | `1 / (1 + e⁻ˣ)`                     | ]0, 1[          | Sortie binaire                 |
| Tanh       | `(eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)`          | ]-1, 1[         | Couches cachées (NLP, RNN)     |
| Softmax    | `eˣⁱ / Σeˣʲ`                        | ]0, 1[ (somme=1)| Sortie multiclasse             |
| LeakyReLU  | `x si x > 0, αx sinon`              | ]-∞, +∞[        | Variante robuste de ReLU       |

---

### Propagation (forward pass)

La propagation est le passage **de l'entrée vers la sortie** du réseau. Pour chaque couche, chaque neurone calcule la somme pondérée de ses entrées, y ajoute son biais, puis applique sa fonction d'activation :

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
sortie = f(z)
```

Ce processus se répète couche par couche jusqu'à produire une prédiction finale. C'est l'étape d'**inférence** : le réseau produit une réponse à partir d'une entrée donnée.

---

### Rétropropagation (backpropagation)

La rétropropagation est l'algorithme qui permet au réseau d'**apprendre** en ajustant ses poids. Une fois la propagation effectuée et l'erreur calculée, on propage cette erreur **en sens inverse** (de la sortie vers l'entrée) en appliquant la règle de dérivation en chaîne :

```
∂L/∂w = ∂L/∂sortie × ∂sortie/∂z × ∂z/∂w
```

Cela donne le **gradient** de chaque poids — c'est-à-dire dans quelle direction et de combien il faut le modifier pour réduire l'erreur. Les poids sont ensuite mis à jour via la descente de gradient.

> Résumé : propagation → calcul de l'erreur → rétropropagation → mise à jour des poids. Ce cycle se répète pour chaque batch d'exemples.

---

### Fonction de perte (loss function)

La fonction de perte mesure **l'écart entre la prédiction du modèle et la valeur réelle**. C'est le signal d'erreur que le réseau cherche à minimiser pendant l'entraînement.

| Tâche                      | Fonction de perte           | Formule simplifiée                        |
|----------------------------|-----------------------------|-------------------------------------------|
| Régression                 | MSE (Mean Squared Error)    | `(1/n) × Σ(ŷ - y)²`                      |
| Régression (robuste)       | MAE (Mean Absolute Error)   | `(1/n) × Σ|ŷ - y|`                       |
| Classification binaire     | Entropie croisée binaire    | `-[y·log(ŷ) + (1-y)·log(1-ŷ)]`          |
| Classification multiclasse | Entropie croisée catégorielle | `-Σ yᵢ·log(ŷᵢ)`                        |

Plus la perte est faible, meilleures sont les prédictions. La rétropropagation calcule le gradient de cette perte par rapport à chaque poids.

---

### Descente de gradient (gradient descent)

La descente de gradient est l'algorithme d'optimisation qui **met à jour les poids** du réseau pour minimiser la fonction de perte. À chaque étape, chaque poids est déplacé dans la direction opposée à son gradient :

```
w ← w - η × ∂L/∂w
```

où `η` (eta) est le **taux d'apprentissage** (learning rate), qui contrôle la taille du pas.

Il en existe plusieurs variantes :

| Variante         | Description                                                              |
|------------------|--------------------------------------------------------------------------|
| Batch GD         | Calcule le gradient sur tout le dataset — lent mais stable               |
| SGD              | Calcule le gradient sur un seul exemple — rapide mais bruité             |
| Mini-batch GD    | Compromis : gradient sur un sous-ensemble (batch) — le plus utilisé      |
| Adam             | Adapte le taux d'apprentissage par paramètre — robuste et efficace       |
| RMSProp          | Variante adaptative, bien adaptée aux séquences et aux RNN               |

---

### Vanishing gradients (disparition du gradient)

Le vanishing gradient est un problème qui survient lors de la rétropropagation dans les réseaux profonds : les gradients deviennent **exponentiellement petits** au fur et à mesure qu'ils remontent vers les premières couches.

**Cause** : certaines fonctions d'activation comme sigmoid et tanh "écrasent" leurs entrées dans une plage étroite (]0,1[ ou ]-1,1[). Leurs dérivées sont très proches de 0 en dehors d'une zone centrale. En multipliant ces petites dérivées couche après couche (règle de la chaîne), le gradient devient négligeable — les premières couches n'apprennent pratiquement plus.

**Conséquences** :
- Les couches proches de l'entrée convergent très lentement ou pas du tout
- Le réseau n'apprend que les dernières couches
- L'entraînement stagne malgré de nombreuses epochs

**Solutions courantes** :

| Solution                  | Description                                                        |
|---------------------------|--------------------------------------------------------------------|
| ReLU                      | Dérivée constante (1 si x > 0), ne sature pas côté positif        |
| Batch Normalization       | Normalise les activations entre les couches                        |
| Initialisation adaptée    | Xavier/Glorot pour tanh, He pour ReLU                             |
| Residual connections      | Connexions "raccourcies" qui court-circuitent certaines couches    |
| Gradient clipping         | Plafonne la norme du gradient pour éviter les exploding gradients  |

## Fonctions d'activation

Une fonction d'activation est une transformation mathématique appliquée à la sortie de chaque neurone. Son rôle est d'introduire de la **non-linéarité** dans le réseau — sans elle, peu importe le nombre de couches, le réseau ne pourrait apprendre que des relations linéaires simples, comme une simple régression.

---

### ReLU — Rectified Linear Unit

```
f(x) = max(0, x)
```

C'est la fonction la plus utilisée dans les couches cachées. Elle laisse passer les valeurs positives telles quelles et annule les valeurs négatives. Sa dérivée est constante (0 ou 1), ce qui évite la disparition du gradient et rend l'entraînement rapide. Son seul défaut : les neurones peuvent "mourir" (toujours à 0) si tous leurs inputs sont négatifs — c'est le phénomène du *dying ReLU*.

---

### Sigmoid

```
f(x) = 1 / (1 + e⁻ˣ)
```

Elle écrase toute valeur réelle dans l'intervalle ]0, 1[, ce qui la rend naturellement interprétable comme une probabilité. Elle est donc le choix standard pour la couche de sortie d'une classification binaire. En revanche, ses dérivées sont très proches de 0 aux extrémités (saturation), ce qui provoque la disparition du gradient dans les couches profondes — d'où son abandon au profit de ReLU dans les couches cachées.

---

### Tanh — Tangente hyperbolique

```
f(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

Similaire à sigmoid mais centrée sur 0, avec une sortie dans ]-1, 1[. Le fait d'être centrée en 0 est un avantage : les gradients oscillent moins pendant l'entraînement. Elle reste populaire dans les réseaux récurrents (RNN, LSTM) mais souffre aussi de saturation aux extrémités.

---

### Leaky ReLU

```
f(x) = x      si x > 0
f(x) = α * x  sinon  (α ≈ 0.01 à 0.1)
```

Une variante de ReLU qui résout le problème du *dying ReLU* : au lieu d'annuler complètement les valeurs négatives, elle leur applique une petite pente `α`. Ainsi, les neurones reçoivent toujours un signal, même pour des entrées négatives.

---

### Softmax

```
f(xᵢ) = eˣⁱ / Σeˣʲ
```

Cas particulier réservé à la couche de sortie multiclasse. Elle prend un vecteur de N valeurs et le transforme en une distribution de probabilités dont la somme vaut exactement 1. La classe prédite est celle qui reçoit la probabilité la plus élevée.

---

### Récapitulatif

| Fonction   | Plage de sortie  | Avantage principal               | Limite principale           | Usage typique            |
|------------|------------------|----------------------------------|-----------------------------|--------------------------|
| ReLU       | [0, +∞[          | Rapide, pas de saturation        | Dying ReLU                  | Couches cachées (défaut) |
| Sigmoid    | ]0, 1[           | Interprétable comme probabilité  | Saturation, vanishing grad  | Sortie binaire           |
| Tanh       | ]-1, 1[          | Centrée en 0                     | Saturation aux extrémités   | RNN, LSTM                |
| Leaky ReLU | ]-∞, +∞[         | Évite le dying ReLU              | α à choisir manuellement    | Couches cachées          |
| Softmax    | ]0, 1[ (Σ=1)     | Distribution de probabilités     | Coûteuse sur grand N        | Sortie multiclasse       | 

## Epochs, Iterations et Batch size

Ces trois notions décrivent comment le dataset est consommé pendant l'entraînement.

- **Batch size** : nombre d'exemples traités en une seule passe avant de mettre à jour les poids. Un batch de 32 signifie que le réseau calcule l'erreur sur 32 exemples, puis effectue une mise à jour.
- **Iteration** (ou step) : une mise à jour des poids = une iteration. Si le dataset contient 1 000 exemples et que le batch size est 32, une epoch contient ⌈1000/32⌉ = 32 iterations.
- **Epoch** : un passage complet sur l'ensemble du dataset d'entraînement. À la fin d'une epoch, chaque exemple a été vu exactement une fois.

```
Nombre d'iterations par epoch = taille du dataset / batch size

Exemple : 10 000 exemples, batch size 100 → 100 iterations par epoch
```

| Notion      | Définition                                      | Unité         |
|-------------|--------------------------------------------------|---------------|
| Batch size  | Nb d'exemples par mise à jour                    | Exemples      |
| Iteration   | Une mise à jour des poids                        | Passe          |
| Epoch       | Un passage complet sur le dataset                | Cycle complet |

---

## Learning rate

Le learning rate (`η`) contrôle la taille du pas effectué lors de chaque mise à jour des poids :

```
w ← w - η × ∂L/∂w
```

C'est l'hyperparamètre le plus critique de l'entraînement.

### Conséquences d'un learning rate trop bas

- La convergence est très lente : il faut beaucoup plus d'epochs pour atteindre un bon résultat
- Le modèle risque de rester bloqué dans un minimum local
- Le temps d'entraînement explose

### Conséquences d'un learning rate trop élevé

- Les poids oscillent fortement autour du minimum sans jamais converger
- La perte peut diverger (augmenter au lieu de diminuer)
- Le modèle n'apprend pas de manière stable

### Bonne pratique

La valeur `0.001` est un point de départ courant. On utilise souvent un **learning rate scheduler** qui réduit progressivement η au cours de l'entraînement, ou des optimiseurs adaptatifs comme Adam qui ajustent η automatiquement par paramètre.

---

## Batch Normalization

La Batch Normalization (BN) est une technique qui **normalise les activations** d'une couche entre chaque mini-batch pendant l'entraînement. Pour chaque batch, elle centre les activations (moyenne ≈ 0) et les réduit (variance ≈ 1), puis applique deux paramètres appris γ et β pour restaurer la capacité expressive du réseau :

```
x̂ = (x - μ_batch) / √(σ²_batch + ε)
sortie = γ × x̂ + β
```

### Pourquoi l'utiliser ?

- **Stabilise l'entraînement** : réduit le phénomène de *internal covariate shift* (changement de distribution des activations d'une couche à l'autre au fil des mises à jour)
- **Accélère la convergence** : permet d'utiliser des learning rates plus élevés
- **Effet régularisant** : réduit légèrement le surapprentissage, parfois au point de se passer du dropout
- **Réduit la sensibilité à l'initialisation des poids**

La BN est généralement insérée après la couche dense ou convolutive, et avant la fonction d'activation.

---

## Optimiseur Adam

Adam (Adaptive Moment Estimation) est l'algorithme d'optimisation le plus utilisé en deep learning. Il combine deux idées :

- **Momentum** : accumule un historique des gradients passés pour donner de l'élan à la descente (évite les oscillations)
- **RMSProp** : adapte le learning rate de chaque paramètre individuellement en fonction de l'amplitude de ses gradients récents

```
m ← β₁ × m + (1 - β₁) × g        ← moyenne mobile des gradients
v ← β₂ × v + (1 - β₂) × g²       ← moyenne mobile des gradients au carré
w ← w - η × m̂ / (√v̂ + ε)
```

Les valeurs par défaut (`η=0.001`, `β₁=0.9`, `β₂=0.999`) fonctionnent bien dans la grande majorité des cas. Adam converge plus vite que SGD classique et nécessite moins de réglage manuel du learning rate.

---

## Définition simplifiée du Perceptron Multicouches

Un Perceptron Multicouches (MLP) est un réseau de neurones artificiels composé de plusieurs couches de neurones connectées les unes aux autres. Les données entrent par la couche d'entrée, traversent une ou plusieurs couches cachées qui apprennent des représentations abstraites, et ressortent par la couche de sortie sous forme de prédiction. Chaque connexion entre neurones porte un poids qui est ajusté pendant l'entraînement par rétropropagation du gradient, de façon à minimiser l'erreur entre la prédiction et la valeur réelle.

---

## Réseaux de neurones convolutifs (CNN)

Un réseau de neurones convolutif (Convolutional Neural Network, CNN) est une architecture de deep learning spécialement conçue pour traiter des données à structure spatiale, en particulier les **images**. Contrairement au MLP, il exploite la structure locale des données grâce à des opérations de convolution qui détectent automatiquement des motifs visuels (contours, textures, formes).

---

## Architecture typique d'un CNN

Un CNN est composé de deux grandes parties :

**1. Partie extraction de caractéristiques**
```
[Entrée image] → [Conv + ReLU] → [Pooling] → [Conv + ReLU] → [Pooling] → ...
```

**2. Partie classification**
```
... → [Flatten] → [Couche dense] → [Softmax] → [Prédiction]
```

Les couches convolutives et de pooling se répètent pour extraire des caractéristiques de plus en plus abstraites, puis la partie dense prend le relais pour la décision finale.

### Hyperparamètres d'un CNN

| Hyperparamètre         | Description                                                       |
|------------------------|-------------------------------------------------------------------|
| Nombre de filtres      | Nb de feature maps produites par chaque couche convolutive        |
| Taille du filtre       | Dimensions du noyau (ex. 3×3, 5×5)                               |
| Stride                 | Pas de déplacement du filtre sur l'image                          |
| Padding                | Ajout de zéros en bordure pour contrôler la taille de sortie      |
| Taille du pooling      | Fenêtre de réduction (ex. 2×2)                                    |
| Nombre de couches      | Profondeur du réseau                                              |
| Learning rate          | Taille du pas de la descente de gradient                          |
| Batch size             | Nombre d'images par mise à jour                                   |
| Dropout                | Taux de neurones désactivés dans la partie dense                  |

---

## Couche convolutive

Une couche convolutive fait glisser un **filtre** (ou noyau) sur l'image d'entrée. À chaque position, elle calcule le produit scalaire entre le filtre et la région de l'image qu'il recouvre, produisant une valeur dans la feature map de sortie.

```
Sortie[i,j] = Σ Σ Entrée[i+m, j+n] × Filtre[m,n] + biais
```

Le filtre se déplace avec un pas (*stride*) et on peut ajouter du *padding* pour conserver les dimensions spatiales. Cette opération est apprise : les valeurs du filtre sont les poids ajustés par rétropropagation.

---

## Filtre de convolution

Un filtre de convolution est une petite matrice de poids (ex. 3×3 ou 5×5) qui détecte un motif spécifique dans l'image : contour horizontal, contour vertical, texture, coin, etc. Un même filtre est appliqué à toute l'image — c'est le principe de **partage des poids**, qui réduit considérablement le nombre de paramètres par rapport à un MLP.

Chaque couche convolutive contient plusieurs filtres, chacun apprenant à détecter un motif différent. Plus on va profond dans le réseau, plus les motifs détectés sont abstraits (passage de contours simples à des formes complexes).

---

## Fonction d'activation d'un CNN

La fonction d'activation utilisée dans les couches convolutives est **ReLU** (`f(x) = max(0, x)`).

### Pourquoi ReLU est la plus adaptée aux CNN ?

- **Efficacité de calcul** : dérivée triviale (0 ou 1), très rapide à évaluer sur des millions de pixels
- **Pas de saturation côté positif** : contrairement à sigmoid ou tanh, ReLU ne compresse pas les grandes valeurs, ce qui préserve l'information lors de la rétropropagation sur des réseaux profonds
- **Sparsité des activations** : en mettant à zéro les valeurs négatives, ReLU crée des représentations creuses, plus efficaces pour encoder des motifs visuels distincts
- **Atténue le vanishing gradient** : sa dérivée constante de 1 pour x > 0 évite la disparition du gradient dans les réseaux profonds

---

## Feature Map

Une feature map (ou carte d'activation) est la sortie produite par l'application d'un filtre de convolution sur l'entrée. Elle représente **la présence et l'intensité d'un motif particulier** à chaque position spatiale de l'image.

- Une couche convolutive avec 32 filtres produit 32 feature maps
- Chaque feature map a les mêmes dimensions spatiales que l'entrée (avec padding) ou légèrement réduites (sans padding)
- Les feature maps des premières couches représentent des motifs simples (contours, couleurs) ; celles des couches profondes représentent des concepts abstraits (yeux, roues, visages)

---

## Couche de Pooling

La couche de pooling réduit les dimensions spatiales des feature maps (largeur × hauteur) tout en conservant les informations les plus importantes. Elle fait glisser une fenêtre (ex. 2×2) sur la feature map avec un stride donné, et applique une opération de réduction à chaque fenêtre.

**Rôles du pooling :**
- Réduire le nombre de paramètres et la charge de calcul
- Introduire une invariance locale aux translations (un motif détecté légèrement décalé donne le même résultat)
- Limiter le surapprentissage

### Types de pooling

**Max Pooling** : retient la valeur maximale de chaque fenêtre. C'est le plus utilisé car il conserve les activations les plus fortes, donc les motifs les plus saillants.
```
[1  3]          Max Pooling 2×2
[2  4]   →   4
```

**Average Pooling** : calcule la moyenne des valeurs de chaque fenêtre. Produit une représentation plus lisse, utilisée parfois dans les dernières couches (ex. Global Average Pooling avant la couche dense).
```
[1  3]          Average Pooling 2×2
[2  4]   →   2.5
```

---

## Couche entièrement connectée (Fully Connected)

La couche entièrement connectée (ou dense) est la dernière partie du CNN. Elle reçoit en entrée les feature maps aplaties en un vecteur 1D (*flatten*) après la dernière couche de pooling.

### Ce qu'elle reçoit

Les feature maps de la dernière couche de pooling sont une représentation compacte et abstraite de l'image d'entrée — un ensemble de caractéristiques de haut niveau extraites par les couches convolutives. L'opération de *flatten* les transforme en un simple vecteur de nombres.

### Son fonctionnement

La couche dense fonctionne exactement comme dans un MLP classique : chaque neurone est connecté à toutes les valeurs du vecteur d'entrée, apprend des combinaisons de caractéristiques, et produit une prédiction. La dernière couche dense utilise une activation **softmax** (multiclasse) ou **sigmoid** (binaire) pour produire des probabilités par classe.

```
Feature maps → Flatten → [Dense + ReLU] → [Dense + Softmax] → Prédiction
```

---

## CNN vs MLP pour la classification d'images

Un CNN est préféré à un MLP dense pour les raisons suivantes :

**Partage des poids** : un filtre convolutif est appliqué à toute l'image avec les mêmes poids. Un MLP dense aurait un poids distinct pour chaque pixel × chaque neurone — une image 224×224×3 représente déjà 150 000 entrées, rendant le réseau ingérable.

**Exploitation de la structure spatiale** : les pixels voisins sont liés entre eux (un contour, une texture). La convolution exploite cette localité ; un MLP traite chaque pixel de manière indépendante et doit tout réapprendre depuis zéro.

**Invariance aux translations** : grâce au pooling, un CNN reconnaît un objet qu'il soit en haut, en bas, à gauche ou à droite de l'image. Un MLP dense est sensible à la position exacte de chaque pixel.

**Hiérarchie de représentations** : les couches successives construisent des abstractions croissantes (pixels → contours → formes → objets), ce qui correspond à la manière dont la vision fonctionne biologiquement.

**Efficacité en paramètres** : un CNN typique pour la classification d'images contient bien moins de paramètres qu'un MLP équivalent, tout en étant bien plus performant.

| Critère                  | MLP dense              | CNN                          |
|--------------------------|------------------------|------------------------------|
| Paramètres               | Très nombreux          | Réduits (partage des poids)  |
| Structure spatiale       | Ignorée                | Exploitée                    |
| Invariance translation   | Non                    | Oui (via pooling)            |
| Performance sur images   | Faible                 | Excellente                   |
| Scalabilité              | Limitée                | Bonne                        |