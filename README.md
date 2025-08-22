# StereoMeasure

Outil complet pour mesurer des objets dans des vidéos stéréoscopiques (vue gauche / vue droite) :
1. Annotation semi-automatique via une interface Qt
2. Export d'un CSV des coordonnées rectangulaires par frame
3. Conversion en mesures physiques (longueur, largeur, profondeur) via `convert_measures.py`

Le workflow est maintenant simplifié : UN CLIC sur l'objet dans l'image gauche déclenche segmentation + matching + boîte droite + écriture de la ligne.

---
## ⚡ Résumé rapide (TL;DR)

```bash
# 1. Créer l'environnement (si pas déjà fait)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Lancer le GUI
python video_matcher.gui.py

# 3. Charger les deux vidéos (gauche puis droite)
# 4. Aller à la frame de début (slider)
# 5. Cliquer sur l'objet dans l'image gauche → une ligne apparaît dans la table
# 6. Bouton "Lancer sur les frames suivantes" → choisir frame de fin + pas
# 7. Bouton "Exporter CSV" (ex: mesures_atimaono.csv)

# 8. Lancer la conversion
python convert_measures.py  # adapter input_filename & baseline avant
```

---
## 🧩 Installation

### Option 1 : Script automatique
```bash
bash install.sh
```

### Option 2 : Manuel
```bash
git clone <repo>
cd stereo_measure
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---
## 🎬 Utilisation de l'interface (`video_matcher.gui.py`)

### 1. Chargement des vidéos
Boutons pour ouvrir la vidéo gauche puis la droite. Les frames sont synchronisées. Le slider permet d'aller directement vers la frame où l'objet commence à être visible.

### 2. Sélection initiale (clic)
Dans l'image gauche : cliquer UNE FOIS sur l'objet à suivre.
Automatiquement le logiciel :
- segmente l'objet (masque gauche)
- recherche la zone correspondante à droite
- dessine les deux boîtes (gauche/droite)
- remplit la ligne dans le tableau (frame + 8 points gauche + 8 points droite)

Si la détection droite échoue, les colonnes R_* restent vides pour cette frame.

### 3. Traitement en séquence
Bouton : **"Lancer sur les frames suivantes"**
- Demande : frame de fin
- Demande : pas (1 = toutes les frames, 2 = 1 sur 2, etc.)
- Affiche une barre de progression + bouton d'annulation.
Chaque frame traitée ajoute/complète une ligne.

### 4. Export
Bouton **"Exporter CSV"** → choisir un nom (ex: `mesures_atimaono.csv`).
Zéros inutiles sont vidés pour alléger le fichier.

### 5. Reset (optionnel)
Bouton **Reset** pour effacer la sélection et repartir sur une nouvelle séquence sans recharger les vidéos.

---
## 📄 Structure du CSV exporté

Colonnes principales (ordre) :
```
Frame,
L_x1,L_y1,L_x2,L_y2,L_x3,L_y3,L_x4,L_y4,
R_x1,R_y1,R_x2,R_y2,R_x3,R_y3,R_x4,R_y4
```
Chaque point correspond aux sommets du quadrilatère détecté (ordre imposé par la logique interne). Les colonnes R_* peuvent être vides si l'appariement a échoué.

---
## 📐 Conversion des mesures (`convert_measures.py`)

Avant d'exécuter : ouvrir le fichier et ajuster :
```python
input_filename = "mesures_atimaono.csv"  # CSV exporté
output_suffix = "_cut"                   # suffixe fichiers de sortie
focale_length = 2400                      # en pixels
baseline = 25                             # en cm (ex: 25 ou 26.5)
```

Puis lancer :
```bash
python convert_measures.py
```
Le script :
1. Calcule milieux des côtés
2. Identifie longueur / largeur (max / min des diagonales de milieux)
3. Calcule disparités (X gauche - X droite)
4. Convertit en coordonnées 3D
5. Produit longueur_cm, largeur_cm, profondeur_m
6. Filtre outliers (IQR) pour moyennes propres
7. Sauvegarde : `mesures_atimaono_cut.csv` + `mesures_atimaono_cut.png`

### Fichiers générés
```
mesures_<site>.csv          # export GUI
mesures_<site>_cut.csv      # enrichi + mesures finales
mesures_<site>_cut.png      # graphiques
```

Graphiques :
- Profondeur (m)
- Longueur (cm) + baseline + moyenne
- Largeur (cm) + moyenne

---
## 🔧 Paramètres de calibration

| Paramètre | Description |
|-----------|-------------|
| focale_length | Focale en pixels (issue de la calibration caméra) |
| baseline | Distance entre optiques (cm) – dépend du montage (ex: 25 ou 26.5) |
| cx, cy | Centre optique supposé (par défaut largeur/2, hauteur/2) |

Adapter `baseline` selon le site (ex: Atimaono vs Méridien). Garder cohérence entre différentes sessions.

---
## 📦 Dépendances principales
Voir `requirements.txt` (extraits) :
- PyQt5
- OpenCV
- numpy
- pandas
- matplotlib

---
## 🧪 Vérification rapide
Après export GUI :
```bash
head -n 3 mesures_atimaono.csv
python convert_measures.py
ls -1 *atimaono_cut*
```

---
## ❓ FAQ rapide
**Les colonnes R_* sont vides** : matching impossible → réessayer depuis une frame plus nette ou recliquer l'objet.

**Le script de conversion affiche des disparités très petites** : vérifier que les points gauche/droite ne sont pas inversés ou baseline incorrecte.

**Graphique plat** : peut indiquer que la segmentation trouve toujours la même zone (objet statique) ou que le pas est trop grand.

---
## 🗂 Organisation des mesures par site
Exemple de fichiers présents :
```
mesures_atimaono.csv
mesures_atimaono_cut.csv
mesures_meridien.csv
mesures_meridien_cut.csv
```
Changer `input_filename` pour traiter l'un ou l'autre.

---
## 🔍 Détails algorithmiques (résumé)
- Extraction SIFT locale
- Filtrage des matches
- Segmentation masque gauche
- Boîte droite par transfert / recherche locale
- Validation rudimentaire (IoU / ratio surfaces)

---
## 🤝 Contributions
1. Créer une branche
2. Faire des commits clairs
3. Ouvrir une Pull Request

---
## 📄 Licence
Logiciel distribué sous licence libre (voir fichier LICENSE si présent).