#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversor Automático de NetCDF para CSV
========================================

Converte automaticamente todos os arquivos .nc de data/modis e data/swot
para pastas modisCSV e swotCSV.

Uso:
    python converter_nc_automatico.py

Autor: FinShark Team
Data: 2024
"""

import glob
import os

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

PASTA_MODIS_ENTRADA = "data/modis"
PASTA_SWOT_ENTRADA = "data/swot"
PASTA_MODIS_SAIDA = "modisCSV"
PASTA_SWOT_SAIDA = "swotCSV"


# =============================================================================
# FUNÇÕES DE CONVERSÃO
# =============================================================================


def converter_bin_para_lat_lon(bin_nums: np.ndarray) -> tuple:
    """
    Converte números de bins MODIS para coordenadas lat/lon.

    Args:
        bin_nums: Array de números de bins

    Returns:
        tuple: (latitudes, longitudes)
    """
    num_rows = 2160
    num_cols = 4320

    row = (bin_nums - 1) // num_cols
    col = (bin_nums - 1) % num_cols

    lat = 90.0 - (row + 0.5) * (180.0 / num_rows)
    lon = (col + 0.5) * (360.0 / num_cols) - 180.0

    return lat, lon


def converter_swot_para_csv(arquivo_nc: str, arquivo_csv: str) -> bool:
    """
    Converte arquivo SWOT NetCDF para CSV.

    Args:
        arquivo_nc: Caminho do arquivo .nc SWOT
        arquivo_csv: Caminho do arquivo CSV de saída

    Returns:
        bool: True se sucesso, False caso contrário
    """
    try:
        # Abrir arquivo NetCDF
        ds = xr.open_dataset(arquivo_nc)

        # Verificar se tem a variável necessária
        if "ssha_karin" not in ds.data_vars:
            ds.close()
            return False

        # Extrair coordenadas
        if "latitude_avg_ssh" in ds.data_vars and "longitude_avg_ssh" in ds.data_vars:
            lats = ds["latitude_avg_ssh"].values
            lons = ds["longitude_avg_ssh"].values
        elif "latitude" in ds.data_vars and "longitude" in ds.data_vars:
            lats = ds["latitude"].values
            lons = ds["longitude"].values
        else:
            ds.close()
            return False

        # Extrair dados de altura da superfície do mar
        ssha = ds["ssha_karin"].values

        # Ajustar dimensões se necessário
        if lats.shape != ssha.shape:
            if len(lats.shape) == 1 and len(ssha.shape) == 2:
                lats = np.tile(lats[:, np.newaxis], (1, ssha.shape[1]))
                lons = np.tile(lons[:, np.newaxis], (1, ssha.shape[1]))

        # Flatten arrays
        lats_flat = lats.flatten()
        lons_flat = lons.flatten()
        ssha_flat = ssha.flatten()

        # Filtrar valores válidos (remover NaN)
        valid_mask = ~(np.isnan(lats_flat) | np.isnan(lons_flat) | np.isnan(ssha_flat))

        # Criar DataFrame
        df = pd.DataFrame(
            {
                "lat": lats_flat[valid_mask],
                "lon": lons_flat[valid_mask],
                "ssha": ssha_flat[valid_mask],
            }
        )

        # Salvar CSV
        df.to_csv(arquivo_csv, index=False)

        ds.close()
        return True

    except Exception as e:
        print(f"      ERRO: {e}")
        return False


def converter_modis_para_csv(arquivo_nc: str, arquivo_csv: str) -> bool:
    """
    Converte arquivo MODIS NetCDF para CSV.

    Args:
        arquivo_nc: Caminho do arquivo .nc MODIS
        arquivo_csv: Caminho do arquivo CSV de saída

    Returns:
        bool: True se sucesso, False caso contrário
    """
    try:
        # Abrir arquivo NetCDF com grupo específico do MODIS
        ds = xr.open_dataset(arquivo_nc, group="level-3_binned_data")

        # Verificar variáveis necessárias
        if "BinList" not in ds.data_vars or "chlor_a" not in ds.data_vars:
            ds.close()
            return False

        # Extrair dados
        binlist = ds["BinList"].values

        # Verificar estrutura de dados
        if not hasattr(binlist, "dtype") or "bin_num" not in binlist.dtype.names:
            ds.close()
            return False

        # Extrair números de bins e converter para lat/lon
        bin_nums = binlist["bin_num"]
        lats, lons = converter_bin_para_lat_lon(bin_nums)

        # Processar clorofila
        chlor_values = ds["chlor_a"].values

        if hasattr(chlor_values, "dtype") and "sum" in chlor_values.dtype.names:
            chlor_sum = chlor_values["sum"]
            nobs = binlist["nobs"]
            chlor_a = np.where(nobs > 0, chlor_sum / nobs, np.nan)
        else:
            chlor_a = chlor_values

        # Filtrar valores válidos
        valid_mask = ~np.isnan(chlor_a) & (chlor_a > 0)

        # Criar DataFrame
        df = pd.DataFrame(
            {
                "lat": lats[valid_mask],
                "lon": lons[valid_mask],
                "chlor_a": chlor_a[valid_mask],
            }
        )

        # Salvar CSV
        df.to_csv(arquivo_csv, index=False)

        ds.close()
        return True

    except Exception as e:
        print(f"      ERRO: {e}")
        return False


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================


def main():
    """Função principal para conversão automática."""
    print("=" * 70)
    print("CONVERSOR AUTOMÁTICO DE NETCDF PARA CSV")
    print("=" * 70)
    print()

    # =========================================================================
    # 1. CONVERTER ARQUIVOS SWOT
    # =========================================================================
    print("📡 PROCESSANDO ARQUIVOS SWOT")
    print("-" * 70)

    # Criar pasta de saída
    os.makedirs(PASTA_SWOT_SAIDA, exist_ok=True)

    # Buscar arquivos SWOT
    arquivos_swot = glob.glob(os.path.join(PASTA_SWOT_ENTRADA, "*.nc"))

    if not arquivos_swot:
        print(f"⚠️  Nenhum arquivo .nc encontrado em {PASTA_SWOT_ENTRADA}")
    else:
        print(f"Encontrados {len(arquivos_swot)} arquivos SWOT")
        print()

        sucesso_swot = 0
        falhas_swot = 0

        for arquivo_nc in tqdm(arquivos_swot, desc="Convertendo SWOT"):
            nome_base = os.path.basename(arquivo_nc)
            nome_csv = nome_base.replace(".nc", ".csv")
            arquivo_csv = os.path.join(PASTA_SWOT_SAIDA, nome_csv)

            if converter_swot_para_csv(arquivo_nc, arquivo_csv):
                sucesso_swot += 1
            else:
                falhas_swot += 1

        print()
        print(f"✅ SWOT: {sucesso_swot} convertidos com sucesso")
        if falhas_swot > 0:
            print(f"❌ SWOT: {falhas_swot} falhas")
        print(f"📁 Salvos em: {PASTA_SWOT_SAIDA}/")

    print()

    # =========================================================================
    # 2. CONVERTER ARQUIVOS MODIS
    # =========================================================================
    print("🛰️  PROCESSANDO ARQUIVOS MODIS")
    print("-" * 70)

    # Criar pasta de saída
    os.makedirs(PASTA_MODIS_SAIDA, exist_ok=True)

    # Buscar arquivos MODIS
    arquivos_modis = glob.glob(os.path.join(PASTA_MODIS_ENTRADA, "*.nc"))

    if not arquivos_modis:
        print(f"⚠️  Nenhum arquivo .nc encontrado em {PASTA_MODIS_ENTRADA}")
    else:
        print(f"Encontrados {len(arquivos_modis)} arquivos MODIS")
        print()

        sucesso_modis = 0
        falhas_modis = 0

        for arquivo_nc in tqdm(arquivos_modis, desc="Convertendo MODIS"):
            nome_base = os.path.basename(arquivo_nc)
            nome_csv = nome_base.replace(".nc", ".csv")
            arquivo_csv = os.path.join(PASTA_MODIS_SAIDA, nome_csv)

            if converter_modis_para_csv(arquivo_nc, arquivo_csv):
                sucesso_modis += 1
            else:
                falhas_modis += 1

        print()
        print(f"✅ MODIS: {sucesso_modis} convertidos com sucesso")
        if falhas_modis > 0:
            print(f"❌ MODIS: {falhas_modis} falhas")
        print(f"📁 Salvos em: {PASTA_MODIS_SAIDA}/")

    # =========================================================================
    # 3. RESUMO FINAL
    # =========================================================================
    print()
    print("=" * 70)
    print("RESUMO FINAL")
    print("=" * 70)
    print(f"SWOT:  {sucesso_swot} sucessos, {falhas_swot} falhas")
    print(f"MODIS: {sucesso_modis} sucessos, {falhas_modis} falhas")
    print(f"TOTAL: {sucesso_swot + sucesso_modis} arquivos CSV gerados")
    print()
    print("✨ CONVERSÃO CONCLUÍDA!")
    print("=" * 70)


if __name__ == "__main__":
    main()
