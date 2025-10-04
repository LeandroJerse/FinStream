#!/usr/bin/env python3
"""
Script para criar dados de teste com múltiplos arquivos SWOT e MODIS
Simula a estrutura real onde cada MODIS representa 1 dia e cada SWOT representa ~10 arquivos por dia
"""

import os
import shutil
from datetime import datetime, timedelta

def criar_dados_teste():
    """Cria estrutura de teste com múltiplos arquivos por data"""
    print("Criando dados de teste com múltiplas datas...")
    
    # Criar estrutura de pastas
    os.makedirs("data/swot", exist_ok=True)
    os.makedirs("data/modis", exist_ok=True)
    os.makedirs("tmp_cache", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Limpar dados existentes
    for file in os.listdir("data/swot"):
        os.remove(f"data/swot/{file}")
    for file in os.listdir("data/modis"):
        os.remove(f"data/modis/{file}")
    
    # Dados existentes (serão copiados)
    arquivo_swot_original = "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
    arquivo_modis_original_1 = "AQUA_MODIS.20240101.L3b.DAY.AT202.nc"
    arquivo_modis_original_2 = "AQUA_MODIS.20240101.L3b.DAY.AT203.nc"
    
    # Verificar se arquivos originais existem
    if not os.path.exists(arquivo_swot_original):
        print(f"AVISO: Arquivo {arquivo_swot_original} não encontrado")
        return
    
    if not os.path.exists(arquivo_modis_original_1):
        print(f"AVISO: Arquivo {arquivo_modis_original_1} não encontrado")
        return
    
    # Criar arquivos para múltiplas datas (2024-01-01 a 2024-01-03)
    datas = ["20240101", "20240102", "20240103"]
    
    print(f"Criando arquivos para {len(datas)} datas...")
    
    for data in datas:
        print(f"\nProcessando data: {data}")
        
        # Criar arquivos MODIS para esta data
        modis_1_nome = f"AQUA_MODIS.{data}.L3b.DAY.AT202.nc"
        modis_2_nome = f"AQUA_MODIS.{data}.L3b.DAY.AT203.nc"
        
        if os.path.exists(arquivo_modis_original_1):
            shutil.copy(arquivo_modis_original_1, f"data/modis/{modis_1_nome}")
            print(f"  Criado: {modis_1_nome}")
        
        if os.path.exists(arquivo_modis_original_2):
            shutil.copy(arquivo_modis_original_2, f"data/modis/{modis_2_nome}")
            print(f"  Criado: {modis_2_nome}")
        
        # Criar 10 arquivos SWOT para esta data (simulando a estrutura real)
        for i in range(10):
            hora_inicio = f"{i*2:02d}0000"  # 00:00, 02:00, 04:00, etc.
            hora_fim = f"{(i*2+1):02d}5959"  # 01:59, 03:59, 05:59, etc.
            
            swot_nome = f"SWOT_L2_LR_SSH_Expert_008_{497+i}_{data}T{hora_inicio}_{data}T{hora_fim}_PGC0_01.nc"
            
            if os.path.exists(arquivo_swot_original):
                shutil.copy(arquivo_swot_original, f"data/swot/{swot_nome}")
                print(f"  Criado: {swot_nome}")
    
    print(f"\nEstrutura criada:")
    print(f"data/")
    print(f"|-- swot/ ({len(os.listdir('data/swot'))} arquivos)")
    print(f"|-- modis/ ({len(os.listdir('data/modis'))} arquivos)")
    
    print(f"\nArquivos criados por data:")
    for data in datas:
        swot_count = len([f for f in os.listdir("data/swot") if data in f])
        modis_count = len([f for f in os.listdir("data/modis") if data in f])
        print(f"  {data}: {swot_count} SWOT, {modis_count} MODIS")
    
    print(f"\nAgora execute: python analise_ultra_eficiente.py")

if __name__ == "__main__":
    criar_dados_teste()
