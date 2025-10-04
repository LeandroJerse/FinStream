#!/usr/bin/env python3
"""
Script Automático de União de Dados Oceânicos e Biológicos
============================================================
Une automaticamente dados de Tubarões + SWOT + MODIS por data
Saída: data/dados_unificados_final.csv
"""

import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from tqdm import tqdm

# ==================== CONFIGURAÇÕES ====================
DATA_DIR = "data"
TUBAROES_DIR = f"{DATA_DIR}/analise_diaria"
SWOT_DIR = f"{DATA_DIR}/swot"
MODIS_DIR = f"{DATA_DIR}/modis"
OUTPUT_FILE = f"{DATA_DIR}/dados_unificados_final.csv"
TOLERANCE_DEG = 1.0  # Tolerância espacial em graus

# Criar diretórios se não existirem
os.makedirs(DATA_DIR, exist_ok=True)


# ==================== FUNÇÕES AUXILIARES ====================


def extrair_data_arquivo(filename: str) -> str:
    """
    Extrai data YYYYMMDD do nome do arquivo.

    Exemplos:
        tubaroes_20240101.csv -> 20240101
        AQUA_MODIS.20240101.L3b.DAY.AT202.nc -> 20240101
        SWOT_L2_LR_SSH_Expert_008_497_20240101T000000_20240101T015959_PGC0_01.nc -> 20240101
    """
    match = re.search(r"(\d{8})", filename)
    return match.group(1) if match else None


def descobrir_arquivos_por_data() -> Dict[str, Dict[str, List[str]]]:
    """
    Descobre todos os arquivos e os agrupa por data.

    Returns:
        Dict com estrutura: {
            "20240101": {
                "tubaroes": "path/to/tubaroes_20240101.csv",
                "swot": ["path/to/swot1.nc", "path/to/swot2.nc", ...],
                "modis": ["path/to/modis1.nc", "path/to/modis2.nc", ...]
            },
            ...
        }
    """
    print("\nDESCOBRINDO ARQUIVOS...")

    arquivos_por_data = {}

    # Descobrir arquivos de tubarões
    tubaroes_files = glob.glob(os.path.join(TUBAROES_DIR, "tubaroes_*.csv"))
    print(f"   Tubarões: {len(tubaroes_files)} arquivos")

    for filepath in tubaroes_files:
        data = extrair_data_arquivo(os.path.basename(filepath))
        if data:
            if data not in arquivos_por_data:
                arquivos_por_data[data] = {"tubaroes": None, "swot": [], "modis": []}
            arquivos_por_data[data]["tubaroes"] = filepath

    # Descobrir arquivos SWOT
    swot_files = glob.glob(os.path.join(SWOT_DIR, "*.nc"))
    print(f"   SWOT: {len(swot_files)} arquivos")

    for filepath in swot_files:
        data = extrair_data_arquivo(os.path.basename(filepath))
        if data:
            if data not in arquivos_por_data:
                arquivos_por_data[data] = {"tubaroes": None, "swot": [], "modis": []}
            arquivos_por_data[data]["swot"].append(filepath)

    # Descobrir arquivos MODIS
    modis_files = glob.glob(os.path.join(MODIS_DIR, "*.nc"))
    print(f"   MODIS: {len(modis_files)} arquivos")

    for filepath in modis_files:
        data = extrair_data_arquivo(os.path.basename(filepath))
        if data:
            if data not in arquivos_por_data:
                arquivos_por_data[data] = {"tubaroes": None, "swot": [], "modis": []}
            arquivos_por_data[data]["modis"].append(filepath)

    # Filtrar apenas datas que têm todos os tipos de arquivos
    datas_completas = {
        data: arquivos
        for data, arquivos in arquivos_por_data.items()
        if arquivos["tubaroes"] and arquivos["swot"] and arquivos["modis"]
    }

    print(f"\nDatas completas encontradas: {len(datas_completas)}")
    for data in sorted(datas_completas.keys()):
        print(
            f"   {data}: {len(datas_completas[data]['swot'])} SWOT, "
            f"{len(datas_completas[data]['modis'])} MODIS"
        )

    return datas_completas


def converter_bin_para_lat_lon(bin_nums: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Converte números de bin MODIS para coordenadas lat/lon."""
    num_rows = 2160
    num_cols = 4320

    row = bin_nums // num_cols
    col = bin_nums % num_cols

    lat = 90.0 - (row + 0.5) * (180.0 / num_rows)
    lon = (col + 0.5) * (360.0 / num_cols) - 180.0

    return lat, lon


def carregar_dados_swot(
    swot_files: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega dados SWOT de múltiplos arquivos.

    Returns:
        (lats, lons, ssha_values)
    """
    all_lats = []
    all_lons = []
    all_ssha = []

    for filepath in swot_files:
        try:
            ds = xr.open_dataset(filepath)

            if "ssha_karin" not in ds.data_vars:
                ds.close()
                continue

            # Extrair coordenadas
            if (
                "latitude_avg_ssh" in ds.data_vars
                and "longitude_avg_ssh" in ds.data_vars
            ):
                lats = ds["latitude_avg_ssh"].values
                lons = ds["longitude_avg_ssh"].values
            else:
                ds.close()
                continue

            ssha = ds["ssha_karin"].values

            # Ajustar dimensões se necessário
            if lats.shape != ssha.shape:
                if len(lats.shape) == 1 and len(ssha.shape) == 2:
                    lats = np.tile(lats[:, np.newaxis], (1, ssha.shape[1])).flatten()
                    lons = np.tile(lons[:, np.newaxis], (1, ssha.shape[1])).flatten()
                    ssha = ssha.flatten()

            # Filtrar valores válidos
            valid_mask = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(ssha))

            if np.any(valid_mask):
                all_lats.extend(lats[valid_mask])
                all_lons.extend(lons[valid_mask])
                all_ssha.extend(ssha[valid_mask])

            ds.close()

        except Exception as e:
            print(f"      AVISO: Erro ao ler SWOT {os.path.basename(filepath)}: {e}")
            continue

    return np.array(all_lats), np.array(all_lons), np.array(all_ssha)


def carregar_dados_modis(
    modis_files: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Carrega dados MODIS de múltiplos arquivos.

    Returns:
        (lats, lons, chlor_a_values)
    """
    all_lats = []
    all_lons = []
    all_chlor = []

    for filepath in modis_files:
        try:
            ds = xr.open_dataset(filepath, group="level-3_binned_data")

            if "BinList" not in ds.data_vars or "chlor_a" not in ds.data_vars:
                ds.close()
                continue

            binlist = ds["BinList"].values

            if hasattr(binlist, "dtype") and "bin_num" in binlist.dtype.names:
                bin_nums = binlist["bin_num"]
                lats, lons = converter_bin_para_lat_lon(bin_nums)

                # Processar clorofila
                chlor_values = ds["chlor_a"].values
                if hasattr(chlor_values, "dtype") and "sum" in chlor_values.dtype.names:
                    chlor_sum = chlor_values["sum"]
                    nobs = binlist["nobs"]
                    chlor_media = np.where(nobs > 0, chlor_sum / nobs, np.nan)

                    # Filtrar valores válidos
                    valid_mask = ~np.isnan(chlor_media) & (chlor_media > 0)

                    if np.any(valid_mask):
                        all_lats.extend(lats[valid_mask])
                        all_lons.extend(lons[valid_mask])
                        all_chlor.extend(chlor_media[valid_mask])

            ds.close()

        except Exception as e:
            print(f"      AVISO: Erro ao ler MODIS {os.path.basename(filepath)}: {e}")
            continue

    return np.array(all_lats), np.array(all_lons), np.array(all_chlor)


def processar_data(data: str, arquivos: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Processa todos os dados de uma data específica e retorna DataFrame unificado.

    Args:
        data: Data no formato YYYYMMDD
        arquivos: Dicionário com caminhos dos arquivos

    Returns:
        DataFrame com dados unificados
    """
    print(f"\nPROCESSANDO {data}")

    # 1. Carregar dados dos tubarões
    print(f"   Carregando tubaroes...")
    tubaroes_df = pd.read_csv(arquivos["tubaroes"])
    print(f"      OK: {len(tubaroes_df)} registros")

    # 2. Carregar dados SWOT
    print(f"   Carregando SWOT ({len(arquivos['swot'])} arquivos)...")
    swot_lats, swot_lons, swot_ssha = carregar_dados_swot(arquivos["swot"])

    if len(swot_lats) == 0:
        print(f"      AVISO: Nenhum dado SWOT valido")
        return pd.DataFrame()

    print(f"      OK: {len(swot_lats)} pontos")

    # 3. Carregar dados MODIS
    print(f"   Carregando MODIS ({len(arquivos['modis'])} arquivos)...")
    modis_lats, modis_lons, modis_chlor = carregar_dados_modis(arquivos["modis"])

    if len(modis_lats) == 0:
        print(f"      AVISO: Nenhum dado MODIS valido")
        return pd.DataFrame()

    print(f"      OK: {len(modis_lats)} pontos")

    # 4. Construir árvores KD para busca espacial eficiente
    print(f"   Construindo arvores KD...")
    swot_kdtree = cKDTree(np.column_stack([swot_lats, swot_lons]))
    modis_kdtree = cKDTree(np.column_stack([modis_lats, modis_lons]))

    # 5. Para cada tubarão, buscar dados ambientais mais próximos
    print(f"   Unindo dados...")
    dados_unificados = []

    for _, row in tqdm(
        tubaroes_df.iterrows(),
        total=len(tubaroes_df),
        desc=f"      Processando {data}",
        leave=False,
    ):

        lat_tubarao = row["lat"]
        lon_tubarao = row["lon"]
        query_point = np.array([[lat_tubarao, lon_tubarao]])

        # Buscar SWOT mais próximo
        swot_dist, swot_idx = swot_kdtree.query(query_point, k=1)
        ssha_ambiente = (
            swot_ssha[swot_idx[0]] if swot_dist[0] < TOLERANCE_DEG else np.nan
        )

        # Buscar MODIS mais próximo
        modis_dist, modis_idx = modis_kdtree.query(query_point, k=1)
        chlor_a_ambiente = (
            modis_chlor[modis_idx[0]] if modis_dist[0] < TOLERANCE_DEG else np.nan
        )

        # Criar registro unificado
        registro = {
            "id_tubarao": row["id_tubarao"],
            "tempo": row["tempo"],
            "lat": lat_tubarao,
            "lon": lon_tubarao,
            "ssha_ambiente": ssha_ambiente,
            "chlor_a_ambiente": chlor_a_ambiente,
            "velocidade": row.get("velocidade", np.nan),
            "nivel_fome": row.get(
                "nivel_fome", row.get("fadiga_nutricional", np.nan)
            ),  # Compatibilidade
            "comportamento": row.get("comportamento", ""),
            "p_forrageio": row.get("p_forrageio", np.nan),
        }

        dados_unificados.append(registro)

    df_unificado = pd.DataFrame(dados_unificados)
    print(f"   SUCESSO: {len(df_unificado)} registros unificados")

    return df_unificado


def main():
    """Função principal."""
    print("=" * 70)
    print("UNIAO AUTOMATICA DE DADOS OCEANICOS E BIOLOGICOS")
    print("=" * 70)
    print(f"Diretorios:")
    print(f"   Tubaroes: {TUBAROES_DIR}")
    print(f"   SWOT: {SWOT_DIR}")
    print(f"   MODIS: {MODIS_DIR}")
    print(f"Saida: {OUTPUT_FILE}")
    print(f"Tolerancia espacial: {TOLERANCE_DEG} graus")

    start_time = datetime.now()

    # 1. Descobrir arquivos por data
    arquivos_por_data = descobrir_arquivos_por_data()

    if not arquivos_por_data:
        print("\nERRO: Nenhuma data completa encontrada!")
        print(
            "   Certifique-se de que existem arquivos de tubaroes, SWOT e MODIS para as mesmas datas."
        )
        return

    # 2. Processar cada data
    print("\n" + "=" * 70)
    print("PROCESSANDO DADOS POR DATA")
    print("=" * 70)

    todos_dados = []

    for data in sorted(arquivos_por_data.keys()):
        df_data = processar_data(data, arquivos_por_data[data])

        if len(df_data) > 0:
            todos_dados.append(df_data)

    # 3. Combinar todos os dados
    if not todos_dados:
        print("\nERRO: Nenhum dado foi processado com sucesso!")
        return

    print("\n" + "=" * 70)
    print("SALVANDO DADOS UNIFICADOS")
    print("=" * 70)

    df_final = pd.concat(todos_dados, ignore_index=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    # 4. Estatísticas finais
    tempo_total = datetime.now() - start_time

    print(f"\nSUCESSO! Dados unificados salvos em:")
    print(f"   {OUTPUT_FILE}")
    print(f"\nESTATISTICAS:")
    print(f"   Datas processadas: {len(arquivos_por_data)}")
    print(f"   Total de registros: {len(df_final):,}")
    print(f"   Tubaroes unicos: {df_final['id_tubarao'].nunique()}")
    print(
        f"   Periodo: {min(arquivos_por_data.keys())} a {max(arquivos_por_data.keys())}"
    )
    print(f"   Tempo de execucao: {tempo_total}")

    # Estatísticas de qualidade dos dados
    print(f"\nQUALIDADE DOS DADOS:")
    print(
        f"   SSHA validos: {df_final['ssha_ambiente'].notna().sum():,} "
        f"({df_final['ssha_ambiente'].notna().mean()*100:.1f}%)"
    )
    print(
        f"   Clorofila validos: {df_final['chlor_a_ambiente'].notna().sum():,} "
        f"({df_final['chlor_a_ambiente'].notna().mean()*100:.1f}%)"
    )

    if (
        df_final["ssha_ambiente"].notna().sum() > 0
        and df_final["chlor_a_ambiente"].notna().sum() > 0
    ):
        correlacao = df_final[["ssha_ambiente", "chlor_a_ambiente"]].corr().iloc[0, 1]
        print(f"   Correlacao SSHA-Clorofila: {correlacao:.4f}")

    print("\n" + "=" * 70)
    print("PROCESSAMENTO CONCLUIDO COM SUCESSO!")
    print("=" * 70)


if __name__ == "__main__":
    main()
