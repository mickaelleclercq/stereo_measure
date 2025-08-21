#!/bin/bash

echo "=== Installation de StereoMeasure ==="

# Vérification de Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 n'est pas installé. Veuillez installer Python 3.7+ avant de continuer."
    exit 1
fi

echo "✅ Python 3 détecté : $(python3 --version)"

# Vérification de pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 n'est pas installé. Veuillez installer pip avant de continuer."
    exit 1
fi

echo "✅ pip3 détecté"

# Installation des dépendances
echo "📦 Installation des dépendances Python..."
pip3 install -r requirements.txt

if [ $? -eq 0 ]; then
    echo "✅ Dépendances installées avec succès"
else
    echo "❌ Erreur lors de l'installation des dépendances"
    exit 1
fi

# Vérification du fichier d'exemple
if [ ! -f "mesures2.csv" ]; then
    echo "⚠️  Fichier mesures2.csv non trouvé. Créez votre fichier de données avant d'exécuter le script."
fi

echo ""
echo "🎉 Installation terminée avec succès !"
echo ""
echo "Pour exécuter StereoMeasure :"
echo "1. Placez votre fichier CSV de données dans ce dossier"
echo "2. Modifiez les paramètres de calibration dans csv2.py si nécessaire"
echo "3. Exécutez : python3 csv2.py"
echo ""
echo "Consultez le README.md pour plus d'informations."