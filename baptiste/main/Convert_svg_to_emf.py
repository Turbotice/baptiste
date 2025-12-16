# -*- coding: utf-8 -*-
"""
Created on Tue Nov  4 14:48:01 2025

@author: Banquise
"""



'''VERSION SVG to EMF'''

# import os
# import subprocess
# from pathlib import Path

# # À modifier si Inkscape n'est pas dans le PATH :
# INKSCAPE_PATH = "C:\Program Files\\Inkscape\\bin\\inkscape.exe"

# # Dossier source et dossier de sortie
# DOSSIER_SOURCE = Path("Y:\Banquise\\Baptiste\\Manuscrit_these\\Figures\\fracture_au_labo\\2_meacnisme_de_rupture")   # <--- à adapter
# DOSSIER_RESULTAT = Path("Y:\Banquise\\Baptiste\\Manuscrit_these\\Figures\\fracture_au_labo\\FICHIERS_EMF")     # <--- à adapter

# def convertir_svg_en_emf(fichier_svg: Path, fichier_emf: Path):
#     """Convertit un fichier SVG en EMF via Inkscape."""
#     fichier_emf.parent.mkdir(parents=True, exist_ok=True)
#     try:
#         subprocess.run([
#             INKSCAPE_PATH,
#             str(fichier_svg),
#             "--export-type=emf",
#             f"--export-filename={fichier_emf}"
#         ], check=True)
#         print(f" Converti : {fichier_svg} -> {fichier_emf}")
#     except subprocess.CalledProcessError:
#         print(f" Erreur lors de la conversion : {fichier_svg}")

# def parcourir_et_convertir(dossier_source: Path, dossier_resultat: Path):
#     """Parcourt tous les sous-dossiers et convertit les fichiers SVG."""
#     fichiers_svg = list(dossier_source.rglob("*.svg"))
#     if not fichiers_svg:
#         print(" Aucun fichier SVG trouvé.")
#         return

#     print(f"{len(fichiers_svg)} fichiers SVG trouvés.")
#     for fichier_svg in fichiers_svg:
#         chemin_relatif = fichier_svg.relative_to(dossier_source)
#         fichier_emf = dossier_resultat / chemin_relatif.with_suffix(".emf")
#         convertir_svg_en_emf(fichier_svg, fichier_emf)

# if __name__ == "__main__":
#     parcourir_et_convertir(DOSSIER_SOURCE, DOSSIER_RESULTAT)
#     print("\n Conversion terminée ! Tous les fichiers EMF sont dans :", DOSSIER_RESULTAT)
    

'''VERSION PDF TO EMF'''

import os
import subprocess
from pathlib import Path

# À modifier selon ton installation :
INKSCAPE_PATH = r"C:\Program Files\Inkscape\bin\inkscape.exe"

# Dossiers à adapter
DOSSIER_SOURCE = Path("Y:\Banquise\\Baptiste\\Manuscrit_these\\Figures")    # <-- ton dossier d'entrée
DOSSIER_RESULTAT = Path("Y:\Banquise\\Baptiste\\Manuscrit_these\\Figures_EMF")      # <-- ton dossier de sortie


DOSSIER_SOURCE = Path("Y:\Banquise\\Baptiste\\Soutenance\\A_convertir")

def convertir_pdf_en_emf(fichier_pdf: Path, fichier_emf: Path):
    """Convertit un fichier PDF en EMF via Inkscape."""
    fichier_emf.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            INKSCAPE_PATH,
            str(fichier_pdf),
            "--export-type=emf",
            f"--export-filename={fichier_emf}"
        ], check=True)
        print(f"CONVERTI : {fichier_pdf} -> {fichier_emf}")
    except subprocess.CalledProcessError:
        print(f"ERREUR lors de la conversion : {fichier_pdf}")

def parcourir_et_convertir(dossier_source: Path, dossier_resultat: Path):
    """Parcourt tous les sous-dossiers et convertit les fichiers PDF."""
    fichiers_pdf = list(dossier_source.rglob("*.pdf"))
    if not fichiers_pdf:
        print(" Aucun fichier PDF trouvé.")
        return

    print(f"{len(fichiers_pdf)} fichiers PDF trouvés.\n")
    for fichier_pdf in fichiers_pdf:
        # Chemin relatif pour conserver la structure
        chemin_relatif = fichier_pdf.relative_to(dossier_source)
        fichier_emf = dossier_resultat / chemin_relatif.with_suffix(".emf")
        convertir_pdf_en_emf(fichier_pdf, fichier_emf)

if __name__ == "__main__":
    parcourir_et_convertir(DOSSIER_SOURCE, DOSSIER_RESULTAT)
    print("\nConversion terminée ! Tous les fichiers EMF sont dans :", DOSSIER_RESULTAT)

'''VERSION PDF TO PNG DPI 600'''

import subprocess
from pathlib import Path

# 🔧 À modifier selon ton installation :
INKSCAPE_PATH = "inkscape"  # ou ex: r"C:\Program Files\Inkscape\bin\inkscape.exe"

# 📁 Dossiers à personnaliser :
DOSSIER_SOURCE = Path("Y:\Banquise\\Baptiste\\Manuscrit_these\\Figures")    # <-- ton dossier d'entrée
DOSSIER_RESULTAT = Path("Y:\Banquise\\Baptiste\\Manuscrit_these\\Figures_PNG")      # <-- ton dossier de sortie
DOSSIER_SOURCE = Path("Y:\Banquise\\Baptiste\\Soutenance\\A_convertir")

# 📏 Résolution d'export en DPI
DPI_EXPORT = 900  # <-- modifie ici selon ton besoin (ex. 300, 600, etc.)

def convertir_pdf_en_png(fichier_pdf: Path, fichier_png: Path, dpi: int = DPI_EXPORT):
    """Convertit un fichier PDF en PNG via Inkscape à la résolution spécifiée."""
    fichier_png.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([
            INKSCAPE_PATH,
            str(fichier_pdf),
            "--export-type=png",
            f"--export-filename={fichier_png}",
            f"--export-dpi={dpi}"
        ], check=True)
        print(f"CONVERTI {fichier_pdf} -> {fichier_png} ({dpi} DPI)")
    except subprocess.CalledProcessError:
        print(f"ERREUR lors de la conversion : {fichier_pdf}")

def parcourir_et_convertir(dossier_source: Path, dossier_resultat: Path, dpi: int = DPI_EXPORT):
    """Parcourt tous les sous-dossiers et convertit les fichiers PDF."""
    fichiers_pdf = list(dossier_source.rglob("*.pdf"))
    if not fichiers_pdf:
        print("Aucun fichier PDF trouvé.")
        return

    print(f"{len(fichiers_pdf)} fichiers PDF trouvés.\n")
    for fichier_pdf in fichiers_pdf:
        chemin_relatif = fichier_pdf.relative_to(dossier_source)
        fichier_png = dossier_resultat / chemin_relatif.with_suffix(".png")
        convertir_pdf_en_png(fichier_pdf, fichier_png, dpi)

if __name__ == "__main__":
    parcourir_et_convertir(DOSSIER_SOURCE, DOSSIER_RESULTAT)
    print("\n Conversion terminée ! Tous les fichiers PNG sont dans :", DOSSIER_RESULTAT)
