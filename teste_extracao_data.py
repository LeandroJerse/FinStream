#!/usr/bin/env python3
"""
Teste da extração de data dos nomes dos arquivos
"""

import re
import os

def extract_date_from_filename(filename: str) -> str:
    """Extrai data do nome do arquivo (SWOT ou MODIS)"""
    try:
        basename = os.path.basename(filename)
        
        # Padrão MODIS: AQUA_MODIS.20240101.L3b.DAY.AT202.nc
        if "MODIS" in basename:
            parts = basename.split(".")
            if len(parts) >= 2:
                return parts[1]  # 20240101
        
        # Padrão SWOT: SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PGC0_01.nc
        elif "SWOT" in basename:
            # Procurar padrão YYYYMMDD no nome
            match = re.search(r'(\d{8})', basename)
            if match:
                return match.group(1)  # 20240101
        
        return "unknown"
    except:
        return "unknown"

def testar_extracao():
    """Testa extração de data em diferentes formatos de arquivo"""
    
    # Arquivos de teste
    arquivos_teste = [
        "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PGC0_01.nc",
        "SWOT_L2_LR_SSH_Expert_008_527_20240102T015030_20240102T024158_PGC0_01.nc",
        "AQUA_MODIS.20240101.L3b.DAY.AT202.nc",
        "AQUA_MODIS.20240102.L3b.DAY.AT203.nc",
        "TERRA_MODIS.20240103.L3b.DAY.AT204.nc"
    ]
    
    print("=== TESTE DE EXTRAÇÃO DE DATA ===")
    
    for arquivo in arquivos_teste:
        data = extract_date_from_filename(arquivo)
        print(f"Arquivo: {arquivo}")
        print(f"Data extraída: {data}")
        print()

if __name__ == "__main__":
    testar_extracao()
