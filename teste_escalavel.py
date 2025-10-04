#!/usr/bin/env python3
"""
Script de teste para a análise escalável
Move os arquivos existentes para a estrutura de pastas esperada
"""

import os
import shutil
from pathlib import Path


def setup_test_environment():
    """Configura ambiente de teste movendo arquivos para estrutura esperada"""
    print("Configurando ambiente de teste...")

    # Criar estrutura de pastas
    os.makedirs("data/swot", exist_ok=True)
    os.makedirs("data/modis", exist_ok=True)
    os.makedirs("tmp_cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Mover arquivos SWOT
    swot_files = [
        "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
    ]

    for file in swot_files:
        if os.path.exists(file):
            shutil.copy(file, f"data/swot/{file}")
            print(f"Copiado: {file} -> data/swot/")

    # Mover arquivos MODIS
    modis_files = [
        "AQUA_MODIS.20240101.L3b.DAY.AT202.nc",
        "AQUA_MODIS.20240101.L3b.DAY.AT203.nc",
    ]

    for file in modis_files:
        if os.path.exists(file):
            shutil.copy(file, f"data/modis/{file}")
            print(f"Copiado: {file} -> data/modis/")

    print("\nEstrutura criada:")
    print("data/")
    print("|-- swot/")
    for file in os.listdir("data/swot"):
        print(f"|   |-- {file}")
    print("|-- modis/")
    for file in os.listdir("data/modis"):
        print(f"    |-- {file}")

    print("\nAgora execute: python analise_ultra_eficiente.py")


if __name__ == "__main__":
    setup_test_environment()
