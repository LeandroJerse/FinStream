#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conversor de Arquivos NetCDF para CSV
======================================

Converte arquivos .nc (SWOT e MODIS) para formato CSV.

Uso:
    python converter_nc_para_csv.py <arquivo.nc> [--output arquivo.csv]
    python converter_nc_para_csv.py --all-swot
    python converter_nc_para_csv.py --all-modis
    python converter_nc_para_csv.py --all

Autor: FinShark Team
Data: 2024
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


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


def converter_swot_para_csv(arquivo_nc: str, arquivo_csv: str = None) -> bool:
    """
    Converte arquivo SWOT NetCDF para CSV.

    Args:
        arquivo_nc: Caminho do arquivo .nc SWOT
        arquivo_csv: Caminho do arquivo CSV de saída (opcional)

    Returns:
        bool: True se sucesso, False caso contrário
    """
    if arquivo_csv is None:
        arquivo_csv = arquivo_nc.replace(".nc", ".csv")

    print(f"Convertendo SWOT: {os.path.basename(arquivo_nc)}")

    try:
        # Abrir arquivo NetCDF
        ds = xr.open_dataset(arquivo_nc)

        # Verificar se tem a variável necessária
        if "ssha_karin" not in ds.data_vars:
            print(f"  ❌ ERRO: Variável 'ssha_karin' não encontrada")
            ds.close()
            return False

        # Extrair coordenadas
        if (
            "latitude_avg_ssh" in ds.data_vars
            and "longitude_avg_ssh" in ds.data_vars
        ):
            lats = ds["latitude_avg_ssh"].values
            lons = ds["longitude_avg_ssh"].values
        elif "latitude" in ds.data_vars and "longitude" in ds.data_vars:
            lats = ds["latitude"].values
            lons = ds["longitude"].values
        else:
            print(f"  ❌ ERRO: Coordenadas não encontradas")
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
        valid_mask = ~(
            np.isnan(lats_flat) | np.isnan(lons_flat) | np.isnan(ssha_flat)
        )

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

        print(f"  ✅ Convertido: {len(df):,} pontos válidos")
        print(f"  📄 Salvo em: {arquivo_csv}")

        ds.close()
        return True

    except Exception as e:
        print(f"  ❌ ERRO: {e}")
        return False


def converter_modis_para_csv(arquivo_nc: str, arquivo_csv: str = None) -> bool:
    """
    Converte arquivo MODIS NetCDF para CSV.

    Args:
        arquivo_nc: Caminho do arquivo .nc MODIS
        arquivo_csv: Caminho do arquivo CSV de saída (opcional)

    Returns:
        bool: True se sucesso, False caso contrário
    """
    if arquivo_csv is None:
        arquivo_csv = arquivo_nc.replace(".nc", ".csv")

    print(f"Convertendo MODIS: {os.path.basename(arquivo_nc)}")

    try:
        # Abrir arquivo NetCDF com grupo específico do MODIS
        ds = xr.open_dataset(arquivo_nc, group="level-3_binned_data")

        # Verificar variáveis necessárias
        if "BinList" not in ds.data_vars or "chlor_a" not in ds.data_vars:
            print(f"  ❌ ERRO: Variáveis necessárias não encontradas")
            ds.close()
            return False

        # Extrair dados
        binlist = ds["BinList"].values

        # Verificar estrutura de dados
        if not hasattr(binlist, "dtype") or "bin_num" not in binlist.dtype.names:
            print(f"  ❌ ERRO: Estrutura de dados inesperada")
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

        print(f"  ✅ Convertido: {len(df):,} pontos válidos")
        print(f"  📄 Salvo em: {arquivo_csv}")

        ds.close()
        return True

    except Exception as e:
        print(f"  ❌ ERRO: {e}")
        return False


def detectar_tipo_nc(arquivo_nc: str) -> str:
    """
    Detecta se o arquivo NetCDF é SWOT ou MODIS baseado no nome.

    Args:
        arquivo_nc: Caminho do arquivo

    Returns:
        str: "swot", "modis" ou "unknown"
    """
    basename = os.path.basename(arquivo_nc).lower()

    if "swot" in basename:
        return "swot"
    elif "modis" in basename or "aqua" in basename:
        return "modis"
    else:
        return "unknown"


def converter_arquivo(arquivo_nc: str, arquivo_csv: str = None) -> bool:
    """
    Converte arquivo NetCDF para CSV (detecta tipo automaticamente).

    Args:
        arquivo_nc: Caminho do arquivo .nc
        arquivo_csv: Caminho do arquivo CSV de saída (opcional)

    Returns:
        bool: True se sucesso
    """
    tipo = detectar_tipo_nc(arquivo_nc)

    if tipo == "swot":
        return converter_swot_para_csv(arquivo_nc, arquivo_csv)
    elif tipo == "modis":
        return converter_modis_para_csv(arquivo_nc, arquivo_csv)
    else:
        print(f"⚠️  AVISO: Tipo desconhecido, tentando SWOT primeiro...")
        if converter_swot_para_csv(arquivo_nc, arquivo_csv):
            return True
        print(f"⚠️  Tentando MODIS...")
        return converter_modis_para_csv(arquivo_nc, arquivo_csv)


def converter_todos_swot(diretorio: str = "data/swot") -> None:
    """Converte todos os arquivos SWOT em um diretório."""
    arquivos = glob.glob(os.path.join(diretorio, "*.nc"))

    if not arquivos:
        print(f"❌ Nenhum arquivo .nc encontrado em {diretorio}")
        return

    print(f"\n{'='*60}")
    print(f"CONVERTENDO {len(arquivos)} ARQUIVOS SWOT")
    print(f"{'='*60}\n")

    sucesso = 0
    falhas = 0

    for arquivo in tqdm(arquivos, desc="Convertendo SWOT"):
        output_dir = os.path.join(diretorio, "csv")
        os.makedirs(output_dir, exist_ok=True)

        arquivo_csv = os.path.join(
            output_dir, os.path.basename(arquivo).replace(".nc", ".csv")
        )

        if converter_swot_para_csv(arquivo, arquivo_csv):
            sucesso += 1
        else:
            falhas += 1

    print(f"\n{'='*60}")
    print(f"RESUMO: {sucesso} sucessos, {falhas} falhas")
    print(f"{'='*60}\n")


def converter_todos_modis(diretorio: str = "data/modis") -> None:
    """Converte todos os arquivos MODIS em um diretório."""
    arquivos = glob.glob(os.path.join(diretorio, "*.nc"))

    if not arquivos:
        print(f"❌ Nenhum arquivo .nc encontrado em {diretorio}")
        return

    print(f"\n{'='*60}")
    print(f"CONVERTENDO {len(arquivos)} ARQUIVOS MODIS")
    print(f"{'='*60}\n")

    sucesso = 0
    falhas = 0

    for arquivo in tqdm(arquivos, desc="Convertendo MODIS"):
        output_dir = os.path.join(diretorio, "csv")
        os.makedirs(output_dir, exist_ok=True)

        arquivo_csv = os.path.join(
            output_dir, os.path.basename(arquivo).replace(".nc", ".csv")
        )

        if converter_modis_para_csv(arquivo, arquivo_csv):
            sucesso += 1
        else:
            falhas += 1

    print(f"\n{'='*60}")
    print(f"RESUMO: {sucesso} sucessos, {falhas} falhas")
    print(f"{'='*60}\n")


def main():
    """Função principal."""
    parser = argparse.ArgumentParser(
        description="Converte arquivos NetCDF (.nc) para CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Converter arquivo específico
  python converter_nc_para_csv.py arquivo.nc

  # Converter com nome de saída específico
  python converter_nc_para_csv.py arquivo.nc --output saida.csv

  # Converter todos os arquivos SWOT
  python converter_nc_para_csv.py --all-swot

  # Converter todos os arquivos MODIS
  python converter_nc_para_csv.py --all-modis

  # Converter todos (SWOT + MODIS)
  python converter_nc_para_csv.py --all
        """,
    )

    parser.add_argument(
        "arquivo", nargs="?", help="Arquivo NetCDF (.nc) para converter"
    )

    parser.add_argument(
        "-o", "--output", help="Arquivo CSV de saída (opcional)"
    )

    parser.add_argument(
        "--all-swot", action="store_true", help="Converter todos os arquivos SWOT"
    )

    parser.add_argument(
        "--all-modis",
        action="store_true",
        help="Converter todos os arquivos MODIS",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Converter todos os arquivos (SWOT + MODIS)",
    )

    parser.add_argument(
        "--swot-dir", default="data/swot", help="Diretório dos arquivos SWOT"
    )

    parser.add_argument(
        "--modis-dir", default="data/modis", help="Diretório dos arquivos MODIS"
    )

    args = parser.parse_args()

    # Processar opções
    if args.all:
        converter_todos_swot(args.swot_dir)
        converter_todos_modis(args.modis_dir)

    elif args.all_swot:
        converter_todos_swot(args.swot_dir)

    elif args.all_modis:
        converter_todos_modis(args.modis_dir)

    elif args.arquivo:
        if not os.path.exists(args.arquivo):
            print(f"❌ ERRO: Arquivo não encontrado: {args.arquivo}")
            sys.exit(1)

        sucesso = converter_arquivo(args.arquivo, args.output)
        sys.exit(0 if sucesso else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
