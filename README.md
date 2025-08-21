# StereoMeasure

**StereoMeasure** est un outil complet d'analyse et de visualisation de mesures stéréoscopiques 3D. Il comprend une interface graphique pour l'annotation de vidéos stéréoscopiques et un système de calcul de dimensions d'objets à partir de données de vision stéréoscopique.

## Fonctionnalités

### Interface graphique (video_matcher.gui.py)
- **Visualisation de vidéos stéréoscopiques** : affichage simultané des vues gauche et droite
- **Annotation manuelle** : sélection de rectangles sur les objets d'intérêt
- **Détection de caractéristiques** : algorithmes ORB/SIFT pour le matching de points
- **Segmentation et tracking** : suivi d'objets à travers les frames
- **Export de données** : sauvegarde des coordonnées des rectangles annotés au format CSV

### Analyse de mesures (convert_measures.py)
- **Calcul de mesures 3D** : longueur, largeur et profondeur d'objets
- **Traitement de données stéréoscopiques** : utilise les disparités pour calculer les distances réelles  
- **Détection d'outliers** : supprime automatiquement les valeurs aberrantes pour des moyennes plus précises
- **Visualisation graphique** : génère des graphiques multi-échelles avec lignes de référence
- **Export de données** : sauvegarde les résultats en CSV et PNG

## Structure des données d'entrée

Le fichier CSV d'entrée doit contenir les colonnes suivantes :
- `Frame` : numéro de frame
- `L_x1`, `L_y1` à `L_x4`, `L_y4` : coordonnées des 4 points du rectangle dans l'image gauche
- `R_x1`, `R_y1` à `R_x4`, `R_y4` : coordonnées des 4 points du rectangle dans l'image droite

## Installation

### Méthode automatique

```bash
bash install.sh
```

### Méthode manuelle

1. Clonez ou téléchargez ce projet
2. Installez Python 3.7+ si ce n'est pas déjà fait
3. Installez les dépendances :

```bash
pip install -r requirements.txt
```

## Configuration

Avant d'exécuter le script, modifiez les paramètres de calibration dans `csv2.py` :

```python
# Configuration des fichiers
input_filename = "mesures2.csv"  # Nom de votre fichier d'entrée
output_suffix = "_cut"           # Suffixe pour les fichiers de sortie

# Configuration des paramètres de calibration
focale_length = 2400    # Focale en pixels
baseline = 26.5         # Distance entre les caméras en cm
```

## Utilisation

### Étape 1 : Annotation des vidéos (video_matcher.gui.py)

Lancez l'interface graphique pour annoter vos vidéos stéréoscopiques :

```bash
python video_matcher.gui.py
```

L'interface permet de :
1. Charger des vidéos stéréoscopiques (vues gauche et droite)
2. Naviguer frame par frame dans les vidéos
3. Dessiner des rectangles autour des objets à mesurer
4. Exporter les coordonnées des rectangles au format CSV

### Étape 2 : Calcul des mesures (convert_measures.py)

Une fois les annotations terminées, utilisez le script de calcul des mesures :

```bash
python convert_measures.py
```

Avant d'exécuter le script, modifiez les paramètres de calibration dans le fichier :

```python
# Configuration des fichiers
input_filename = "mesures.csv"  # Fichier CSV généré par l'interface graphique
output_suffix = "_cut"          # Suffixe pour les fichiers de sortie

# Configuration des paramètres de calibration  
focale_length = 2400    # Focale en pixels
baseline = 26.5         # Distance entre les caméras en cm
```

### Fichiers générés

- `mesures2_cut.csv` : fichier CSV avec les mesures calculées
- `mesures2_cut.png` : graphiques de visualisation

### Sortie graphique

Le script génère trois graphiques :

1. **Profondeur** (en mètres) - axe Y bleu
2. **Longueur et Baseline** (en cm) - axe Y vert avec :
   - Courbe verte : longueur mesurée
   - Ligne rouge pointillée : baseline de référence
   - Ligne noire pointillée : moyenne des longueurs (sans outliers)
3. **Largeur** (en cm) - axe Y magenta avec :
   - Courbe magenta : largeur mesurée  
   - Ligne noire pointillée : moyenne des largeurs (sans outliers)

## Principe de fonctionnement

### Calcul des mesures

1. **Points milieux** : calcule les milieux des côtés du rectangle
2. **Distances 2D** : calcule les distances entre milieux opposés
3. **Longueur/Largeur** : identifie la plus grande et plus petite distance
4. **Disparités** : calcule les disparités entre images gauche et droite
5. **Conversion 3D** : utilise la calibration stéréo pour obtenir les mesures réelles

### Formules utilisées

**Profondeur** :
```
Z = (focale_length × baseline) / disparité
```

**Distance 3D** :
```
X = (x - cx) × Z / focale_length
Y = (y - cy) × Z / focale_length
distance = √((X2-X1)² + (Y2-Y1)² + (Z2-Z1)²)
```

### Traitement des outliers

- Utilise la méthode IQR (Interquartile Range)
- Supprime les valeurs en dehors de [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Gère automatiquement les valeurs manquantes (NaN)

## Paramètres de calibration

- **focale_length** : focale de la caméra en pixels (à obtenir via calibration)
- **baseline** : distance entre les deux caméras en cm
- **cx, cy** : centre de l'image (par défaut : 3840/2, 2160/2)

## Dépendances

- pandas >= 1.3.0
- numpy >= 1.21.0  
- matplotlib >= 3.5.0

## Structure du projet

```
StereoMeasure/
├── video_matcher.gui.py  # Interface graphique d'annotation
├── convert_measures.py   # Script de calcul des mesures
├── requirements.txt      # Dépendances Python  
├── install.sh           # Script d'installation
├── README.md            # Documentation
├── mesures.csv          # Fichier d'exemple d'annotations
├── mesures_cut.csv      # Fichier de sortie (généré)
└── mesures_cut.png      # Graphique de sortie (généré)
```

## Exemples de sortie console

```
Chargement du fichier: mesures2.csv
Nombre de lignes chargées: 100
Distance entre les points : 25.34 cm
Profondeur Z: 145.67 cm
...
Moyenne longueur (sans outliers): 24.8 cm
Moyenne largeur (sans outliers): 12.3 cm
Fichier sauvegardé sous: mesures2_cut.csv
Graphique sauvegardé sous: mesures2_cut.png
```

## Contributions

Pour contribuer au projet :
1. Fork le projet
2. Créez une branche pour votre fonctionnalité
3. Commitez vos modifications
4. Poussez vers la branche
5. Ouvrez une Pull Request

## License

Ce projet est sous licence libre. Consultez le fichier LICENSE pour plus de détails.