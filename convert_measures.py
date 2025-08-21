import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# Configuration des fichiers
input_filename = "mesures.csv"
output_suffix = "_cut"

# Génération automatique du nom de sortie
base_name, ext = os.path.splitext(input_filename)
output_filename = f"{base_name}{output_suffix}{ext}"

# Configuration des paramètres de calibration
focale_length = 2400  # Focale en pixels
baseline = 26.5  # cm pour Meridien_test_support_08_2025
# baseline = 25  # cm pour Atimaono_07_2025


def compute_taille(x1, y1, disp1, x2, y2, disp2, baseline, focale_length):
    """Calcule la distance 3D entre deux points à partir de leurs coordonnées et disparités."""
    cx, cy = 3840/2, 2160/2  # Centre de l'image

    # Calcul des profondeurs
    Z1 = (focale_length * baseline) / disp1
    Z2 = (focale_length * baseline) / disp2

    # Conversion en coordonnées 3D
    X1 = (x1 - cx) * Z1 / focale_length
    Y1 = (y1 - cy) * Z1 / focale_length
    X2 = (x2 - cx) * Z2 / focale_length
    Y2 = (y2 - cy) * Z2 / focale_length
    
    distance = ((X2 - X1) ** 2 + (Y2 - Y1) ** 2 + (Z2 - Z1) ** 2) ** 0.5
    print(f"Distance entre les points : {distance:.2f} cm")
    return round(distance, 3)

def get_depth(disp1, disp2, baseline, focale_length):
    """Calcule la profondeur moyenne à partir de deux disparités."""
    Z1 = (focale_length * baseline) / disp1
    Z2 = (focale_length * baseline) / disp2
    depth = (Z1 + Z2) / 2
    print(f"Profondeur Z: {depth:.2f} cm")
    return depth


# Chargement des données
print(f"Chargement du fichier: {input_filename}")
df = pd.read_csv(input_filename)
print(f"Nombre de lignes chargées: {len(df)}")

def milieu(x1, y1, x2, y2):
    """Calcule le point milieu entre deux points."""
    return (x1 + x2) / 2, (y1 + y2) / 2

# Calcul des milieux
for prefix in ["L", "R"]:
    for i in range(1, 4):
        x1, y1 = df[f"{prefix}_x{i}"], df[f"{prefix}_y{i}"]
        x2, y2 = df[f"{prefix}_x{i+1}"], df[f"{prefix}_y{i+1}"]
        df[f"{prefix}_mid{i}_x"], df[f"{prefix}_mid{i}_y"] = milieu(x1, y1, x2, y2)
    x1, y1 = df[f"{prefix}_x4"], df[f"{prefix}_y4"]
    x2, y2 = df[f"{prefix}_x1"], df[f"{prefix}_y1"]
    df[f"{prefix}_mid4_x"], df[f"{prefix}_mid4_y"] = milieu(x1, y1, x2, y2)

# Calcul des distances entre milieux
for prefix in ["L", "R"]:
    df[f"{prefix}_dist_13"] = np.sqrt((df[f"{prefix}_mid3_x"] - df[f"{prefix}_mid1_x"])**2 + (df[f"{prefix}_mid3_y"] - df[f"{prefix}_mid1_y"])**2)
    df[f"{prefix}_dist_24"] = np.sqrt((df[f"{prefix}_mid4_x"] - df[f"{prefix}_mid2_x"])**2 + (df[f"{prefix}_mid4_y"] - df[f"{prefix}_mid2_y"])**2)

# Longueur et largeur
for prefix in ["L", "R"]:
    df[f"{prefix}_longueur"] = df[[f"{prefix}_dist_13", f"{prefix}_dist_24"]].max(axis=1)
    df[f"{prefix}_largeur"]  = df[[f"{prefix}_dist_13", f"{prefix}_dist_24"]].min(axis=1)
    df[f"{prefix}_long_pts"]  = np.where(df[f"{prefix}_dist_13"] >= df[f"{prefix}_dist_24"], "mid1-mid3", "mid2-mid4")
    df[f"{prefix}_large_pts"] = np.where(df[f"{prefix}_dist_13"] < df[f"{prefix}_dist_24"], "mid1-mid3", "mid2-mid4")

# Disparité (différence sur l'axe X uniquement)
for i in range(1,5):
    df[f"disparity_mid{i}"] = df[f"L_mid{i}_x"] - df[f"R_mid{i}_x"]

# Sauvegarde
#df.to_csv("rectangles_milieux_distances_disparity.csv", index=False)

print(df.head())


# Initialisation des listes pour les mesures en cm/m
longueur_cm = []
largeur_cm = []
profondeur_m = []

for idx, row in df.iterrows():
    # --- Longueur ---
    if row["L_long_pts"] == "mid1-mid3":
        x1, y1, disp1 = row["L_mid1_x"], row["L_mid1_y"], row["disparity_mid1"]
        x2, y2, disp2 = row["L_mid3_x"], row["L_mid3_y"], row["disparity_mid3"]
    else:
        x1, y1, disp1 = row["L_mid2_x"], row["L_mid2_y"], row["disparity_mid2"]
        x2, y2, disp2 = row["L_mid4_x"], row["L_mid4_y"], row["disparity_mid4"]
    longueur_cm.append(round(compute_taille(x1, y1, disp1, x2, y2, disp2,baseline,focale_length),3))
    profondeur_m.append(round(get_depth(disp1, disp2, baseline, focale_length)/100,3))
    
    # --- Largeur ---
    if row["L_large_pts"] == "mid1-mid3":
        x1, y1, disp1 = row["L_mid1_x"], row["L_mid1_y"], row["disparity_mid1"]
        x2, y2, disp2 = row["L_mid3_x"], row["L_mid3_y"], row["disparity_mid3"]
    else:
        x1, y1, disp1 = row["L_mid2_x"], row["L_mid2_y"], row["disparity_mid2"]
        x2, y2, disp2 = row["L_mid4_x"], row["L_mid4_y"], row["disparity_mid4"]
    largeur_cm.append(round(compute_taille(x1, y1, disp1, x2, y2, disp2,baseline,focale_length),3))

df["longueur_cm"] = longueur_cm
df["largeur_cm"] = largeur_cm
df["profondeur_m"] = profondeur_m

colonnes_finales = list(df.columns[:17]) + ["L_longueur","L_largeur",
"disparity_mid1","disparity_mid2","disparity_mid3","disparity_mid4",
"longueur_cm","largeur_cm","profondeur_m"]


df_final = df[colonnes_finales]

df_final.to_csv(output_filename, index=False)
print(f"Fichier sauvegardé sous: {output_filename}")

# Calcul des moyennes en excluant les outliers (méthode IQR)
def remove_outliers(data):
    """Supprime les outliers en utilisant la méthode IQR (Interquartile Range)"""
    # Supprime d'abord les valeurs NaN
    data_clean = data.dropna()
    if len(data_clean) == 0:
        return data_clean
    
    Q1 = np.percentile(data_clean, 25)
    Q3 = np.percentile(data_clean, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data_clean[(data_clean >= lower_bound) & (data_clean <= upper_bound)]

# Calcul des moyennes sans outliers
longueur_clean = remove_outliers(df_final['longueur_cm'])
largeur_clean = remove_outliers(df_final['largeur_cm'])
moyenne_longueur = longueur_clean.mean() if len(longueur_clean) > 0 else 0
moyenne_largeur = largeur_clean.mean() if len(largeur_clean) > 0 else 0

print(f"Moyenne longueur (sans outliers): {moyenne_longueur:.2f} cm")
print(f"Moyenne largeur (sans outliers): {moyenne_largeur:.2f} cm")

# Génération du graphique
graph_filename = f"{base_name}{output_suffix}.png"

# Création de la figure avec 3 sous-graphiques
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle(f'Mesures - {base_name}', fontsize=16)

# Graphique 1: Profondeur
ax1.plot(df_final['Frame'], df_final['profondeur_m'], 'b-', marker='o', label='Profondeur (m)')
ax1.set_ylabel('Profondeur (m)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Graphique 2: Longueur et Baseline
ax2.plot(df_final['Frame'], df_final['longueur_cm'], 'g-', marker='s', label='Longueur (cm)')
ax2.axhline(y=baseline, color='r', linestyle='--', linewidth=2, label=f'Baseline ({baseline} cm)')
if moyenne_longueur > 0:
    ax2.axhline(y=moyenne_longueur, color='black', linestyle='--', linewidth=2, label=f'Moyenne longueur ({moyenne_longueur:.1f} cm)')
ax2.set_ylabel('Longueur / Baseline (cm)', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Graphique 3: Largeur
ax3.plot(df_final['Frame'], df_final['largeur_cm'], 'm-', marker='^', label='Largeur (cm)')
if moyenne_largeur > 0:
    ax3.axhline(y=moyenne_largeur, color='black', linestyle='--', linewidth=2, label=f'Moyenne largeur ({moyenne_largeur:.1f} cm)')
ax3.set_ylabel('Largeur (cm)', color='m')
ax3.set_xlabel('Numéro de Frame')
ax3.tick_params(axis='y', labelcolor='m')
ax3.grid(True, alpha=0.3)
ax3.legend()

plt.tight_layout()
plt.savefig(graph_filename, dpi=300, bbox_inches='tight')
plt.show()
print(f"Graphique sauvegardé sous: {graph_filename}")
