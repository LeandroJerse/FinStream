#!/usr/bin/env python3
"""
Análise Ultra-Eficiente e Escalável - SWOT x MODIS x Comportamento Animal
NASA Ocean Data + IA de Comportamento - FinStream Project
Processamento automático de múltiplos arquivos com geração de dataset ML
"""

import glob
import os
import pickle
import warnings
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from tqdm import tqdm

# Configurações globais
TOLERANCE_DEGREES = 1.0
TIME_TOL_HOURS = 12
BATCH_SIZE = 10000
N_JOBS = -1  # Usar todos os cores disponíveis
CACHE_DIR = "tmp_cache"
RESULTS_DIR = "results"
DATA_DIR = "data"
TUBAROES_DIR = f"{DATA_DIR}/analise_diaria"

# Criar diretórios necessários
for dir_name in [
    CACHE_DIR,
    RESULTS_DIR,
    DATA_DIR,
    f"{DATA_DIR}/swot",
    f"{DATA_DIR}/modis",
    TUBAROES_DIR,
]:
    os.makedirs(dir_name, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)


def discover_files(folder: str, pattern: str = "*.nc") -> List[str]:
    """
    Descobre todos os arquivos NetCDF em uma pasta

    Args:
        folder: Caminho da pasta
        pattern: Padrão de busca (padrão: *.nc)

    Returns:
        Lista de caminhos de arquivos encontrados
    """
    if not os.path.exists(folder):
        print(f"AVISO: Pasta {folder} não encontrada. Criando...")
        os.makedirs(folder, exist_ok=True)
        return []

    files = glob.glob(os.path.join(folder, pattern))
    print(f"Encontrados {len(files)} arquivos {pattern} em {folder}")
    return sorted(files)


def discover_tubaroes_files() -> Dict[str, str]:
    """
    Descobre todos os arquivos CSV de tubarões e extrai as datas.

    Returns:
        Dicionário {data: caminho_arquivo}
    """
    if not os.path.exists(TUBAROES_DIR):
        print(f"AVISO: Pasta {TUBAROES_DIR} não encontrada")
        return {}

    files = glob.glob(os.path.join(TUBAROES_DIR, "tubaroes_*.csv"))
    tubaroes_files = {}

    for file_path in files:
        filename = os.path.basename(file_path)
        # Extrair data do formato: tubaroes_YYYYMMDD.csv
        if filename.startswith("tubaroes_") and filename.endswith(".csv"):
            date_str = filename[9:17]  # YYYYMMDD
            try:
                # Validar se é uma data válida
                datetime.strptime(date_str, "%Y%m%d")
                tubaroes_files[date_str] = file_path
            except ValueError:
                print(f"AVISO: Nome de arquivo inválido: {filename}")

    print(f"Encontrados {len(tubaroes_files)} arquivos de tubarões")
    return tubaroes_files


def load_tubaroes_data(
    date_str: str, tubaroes_files: Dict[str, str]
) -> Optional[pd.DataFrame]:
    """
    Carrega dados dos tubarões para uma data específica.

    Args:
        date_str: Data no formato YYYYMMDD
        tubaroes_files: Dicionário de arquivos de tubarões

    Returns:
        DataFrame com dados dos tubarões ou None se não encontrado
    """
    if date_str not in tubaroes_files:
        return None

    try:
        df = pd.read_csv(tubaroes_files[date_str])
        print(f"Dados tubarões carregados: {len(df)} registros para {date_str}")
        return df
    except Exception as e:
        print(f"ERRO ao carregar dados dos tubarões para {date_str}: {e}")
        return None


def create_ml_features_targets(
    tubaroes_df: pd.DataFrame, modis_kdtree_data: Dict
) -> pd.DataFrame:
    """
    Cria features e targets para treinamento de IA de comportamento animal.

    Args:
        tubaroes_df: DataFrame com dados dos tubarões
        modis_kdtree_data: Dados da árvore KD do MODIS para busca espacial

    Returns:
        DataFrame com features e targets para ML
    """
    if tubaroes_df is None or len(tubaroes_df) == 0:
        return pd.DataFrame()

    # Converter tempo para datetime
    tubaroes_df["tempo"] = pd.to_datetime(tubaroes_df["tempo"])

    # Ordenar por tubarão e tempo
    tubaroes_df = tubaroes_df.sort_values(["id_tubarao", "tempo"])

    ml_data = []
    kdtree = modis_kdtree_data["kdtree"]
    modis_data = modis_kdtree_data["data"]

    print(f"Processando {len(tubaroes_df)} registros para features ML...")

    for i, row in tqdm(
        tubaroes_df.iterrows(), total=len(tubaroes_df), desc="Criando features ML"
    ):
        # Features de entrada (estado atual)
        current_lat = row["lat"]
        current_lon = row["lon"]
        current_time = row["tempo"]
        current_ssha = row["ssha"]
        current_chlor_a = row["chlor_a"]
        current_velocidade = row["velocidade"]
        current_fadiga = row["fadiga_nutricional"]
        current_comportamento = row["comportamento"]
        current_p_forrageio = row["p_forrageio"]

        # Extrair hora do dia (feature temporal)
        hora_dia = current_time.hour + current_time.minute / 60.0

        # Buscar dados ambientais mais próximos usando KDTree
        query_point = np.array([[current_lat, current_lon]])
        distances, indices = kdtree.query(query_point, k=1)

        if len(indices) > 0:
            nearest_idx = indices[0]
            if nearest_idx < len(modis_data):
                env_ssha = modis_data[nearest_idx]["ssha"]
                env_chlor_a = modis_data[nearest_idx]["chlor_a"]
            else:
                env_ssha = current_ssha
                env_chlor_a = current_chlor_a
        else:
            env_ssha = current_ssha
            env_chlor_a = current_chlor_a

        # Calcular gradientes espaciais (derivadas)
        gradiente_ssha = 0.0
        gradiente_chlor_a = 0.0

        if len(indices) > 0 and nearest_idx < len(modis_data):
            # Buscar pontos vizinhos para calcular gradiente
            distances_multi, indices_multi = kdtree.query(
                query_point, k=min(5, len(modis_data))
            )
            if len(indices_multi) > 1:
                # Calcular gradiente simples
                ssha_values = [
                    modis_data[idx]["ssha"]
                    for idx in indices_multi
                    if idx < len(modis_data)
                ]
                chlor_values = [
                    modis_data[idx]["chlor_a"]
                    for idx in indices_multi
                    if idx < len(modis_data)
                ]

                if len(ssha_values) > 1:
                    gradiente_ssha = np.std(ssha_values)
                if len(chlor_values) > 1:
                    gradiente_chlor_a = np.std(chlor_values)

        # Buscar próximo registro para criar targets
        next_row = tubaroes_df[
            (tubaroes_df["id_tubarao"] == row["id_tubarao"])
            & (tubaroes_df["tempo"] > current_time)
        ]

        if len(next_row) > 0:
            next_row = next_row.iloc[0]

            # Targets (comportamento futuro)
            target_comportamento = next_row["comportamento"]
            target_p_forrageio = next_row["p_forrageio"]
            target_lat = next_row["lat"]
            target_lon = next_row["lon"]
            target_velocidade = next_row["velocidade"]

            # Calcular distância percorrida
            distancia_percorrida = np.sqrt(
                (target_lat - current_lat) ** 2 + (target_lon - current_lon) ** 2
            )

            # Calcular direção de movimento
            direcao_lat = target_lat - current_lat
            direcao_lon = target_lon - current_lon

            # Features históricas (comportamento anterior)
            prev_rows = tubaroes_df[
                (tubaroes_df["id_tubarao"] == row["id_tubarao"])
                & (tubaroes_df["tempo"] < current_time)
            ]

            comportamento_anterior = "busca"  # default
            velocidade_anterior = 0.0
            fadiga_anterior = 0.0

            if len(prev_rows) > 0:
                last_prev = prev_rows.iloc[-1]
                comportamento_anterior = last_prev["comportamento"]
                velocidade_anterior = last_prev["velocidade"]
                fadiga_anterior = last_prev["fadiga_nutricional"]

            # Criar registro para ML
            ml_record = {
                # Features de entrada (X)
                "lat_atual": current_lat,
                "lon_atual": current_lon,
                "hora_dia": hora_dia,
                "ssha_atual": current_ssha,
                "chlor_a_atual": current_chlor_a,
                "velocidade_atual": current_velocidade,
                "fadiga_atual": current_fadiga,
                "comportamento_atual": current_comportamento,
                "p_forrageio_atual": current_p_forrageio,
                "ssha_ambiente": env_ssha,
                "chlor_a_ambiente": env_chlor_a,
                "gradiente_ssha": gradiente_ssha,
                "gradiente_chlor_a": gradiente_chlor_a,
                "comportamento_anterior": comportamento_anterior,
                "velocidade_anterior": velocidade_anterior,
                "fadiga_anterior": fadiga_anterior,
                # Targets de saída (y)
                "target_comportamento": target_comportamento,
                "target_p_forrageio": target_p_forrageio,
                "target_lat": target_lat,
                "target_lon": target_lon,
                "target_velocidade": target_velocidade,
                "distancia_percorrida": distancia_percorrida,
                "direcao_lat": direcao_lat,
                "direcao_lon": direcao_lon,
                # Metadados
                "id_tubarao": row["id_tubarao"],
                "tempo_atual": current_time,
                "tempo_target": next_row["tempo"],
            }

            ml_data.append(ml_record)

    return pd.DataFrame(ml_data)


def create_ml_features_targets_reais(
    tubaroes_df: pd.DataFrame, modis_kdtree_data: Dict, swot_kdtree_data: Dict
) -> pd.DataFrame:
    """
    Cria features e targets para treinamento de IA usando APENAS dados reais.

    IMPORTANTE: Esta função usa apenas:
    - Posições reais dos tubarões (lat, lon, tempo)
    - Dados ambientais reais dos satélites (SWOT SSHA, MODIS clorofila)
    - NÃO usa dados simulados como velocidade, fadiga_nutricional, comportamento, etc.

    Args:
        tubaroes_df: DataFrame com dados dos tubarões (contém dados reais + simulados)
        modis_kdtree_data: Dados da árvore KD do MODIS para busca espacial
        swot_kdtree_data: Dados da árvore KD do SWOT para busca espacial

    Returns:
        DataFrame com features e targets para ML (apenas dados reais)
    """
    if tubaroes_df is None or len(tubaroes_df) == 0:
        return pd.DataFrame()

    # Converter tempo para datetime
    tubaroes_df["tempo"] = pd.to_datetime(tubaroes_df["tempo"])

    # Ordenar por tubarão e tempo
    tubaroes_df = tubaroes_df.sort_values(["id_tubarao", "tempo"])

    ml_data = []
    modis_kdtree = modis_kdtree_data["kdtree"]
    modis_data = modis_kdtree_data["data"]
    swot_kdtree = swot_kdtree_data["kdtree"]
    swot_data = swot_kdtree_data["data"]

    # Verificar se as árvores KD estão disponíveis
    if modis_kdtree is None or swot_kdtree is None:
        print("AVISO: Árvore KD não disponível, pulando criação de features ML")
        return pd.DataFrame()

    print(
        f"Processando {len(tubaroes_df)} registros para features ML (apenas dados reais)..."
    )
    print(
        "USANDO APENAS: posições (lat/lon/tempo) + dados ambientais reais (SWOT/MODIS)"
    )

    for i, row in tqdm(
        tubaroes_df.iterrows(), total=len(tubaroes_df), desc="Criando features ML"
    ):
        # Features de entrada (APENAS dados reais dos tubarões)
        current_lat = row["lat"]  # REAL: posição do tubarão
        current_lon = row["lon"]  # REAL: posição do tubarão
        current_time = row["tempo"]  # REAL: horário do ping

        # Extrair hora do dia (feature temporal real)
        hora_dia = current_time.hour + current_time.minute / 60.0

        # Buscar dados ambientais REAIS dos satélites usando KDTree
        query_point = np.array([[current_lat, current_lon]])

        # Dados MODIS REAIS (clorofila do satélite)
        modis_distances, modis_indices = modis_kdtree.query(query_point, k=1)
        if len(modis_indices) > 0 and modis_indices[0] < len(modis_data):
            chlor_a_ambiente = modis_data[modis_indices[0]]["chlor_a"]
        else:
            chlor_a_ambiente = 0.0

        # Dados SWOT REAIS (altura da superfície do mar do satélite)
        swot_distances, swot_indices = swot_kdtree.query(query_point, k=1)
        if len(swot_indices) > 0 and swot_indices[0] < len(swot_data):
            ssha_ambiente = swot_data[swot_indices[0]]["ssha"]
        else:
            ssha_ambiente = 0.0

        # Calcular gradientes espaciais (variação do ambiente real)
        gradiente_ssha = 0.0
        gradiente_chlor_a = 0.0

        # Gradiente MODIS (variação espacial real da clorofila)
        if len(modis_indices) > 0:
            modis_distances_multi, modis_indices_multi = modis_kdtree.query(
                query_point, k=min(5, len(modis_data))
            )
            if len(modis_indices_multi) > 1:
                chlor_values = [
                    modis_data[idx]["chlor_a"]
                    for idx in modis_indices_multi
                    if idx < len(modis_data)
                ]
                if len(chlor_values) > 1:
                    gradiente_chlor_a = np.std(chlor_values)

        # Gradiente SWOT (variação espacial real da altura do mar)
        if len(swot_indices) > 0:
            swot_distances_multi, swot_indices_multi = swot_kdtree.query(
                query_point, k=min(5, len(swot_data))
            )
            if len(swot_indices_multi) > 1:
                ssha_values = [
                    swot_data[idx]["ssha"]
                    for idx in swot_indices_multi
                    if idx < len(swot_data)
                ]
                if len(ssha_values) > 1:
                    gradiente_ssha = np.std(ssha_values)

        # Buscar próximo registro para criar targets (próxima posição real)
        next_row = tubaroes_df[
            (tubaroes_df["id_tubarao"] == row["id_tubarao"])
            & (tubaroes_df["tempo"] > current_time)
        ]

        if len(next_row) > 0:
            next_row = next_row.iloc[0]

            # Targets (APENAS dados reais - próxima posição do tubarão)
            target_lat = next_row["lat"]  # REAL: próxima posição
            target_lon = next_row["lon"]  # REAL: próxima posição

            # Calcular distância percorrida (real - baseada em posições reais)
            distancia_percorrida = np.sqrt(
                (target_lat - current_lat) ** 2 + (target_lon - current_lon) ** 2
            )

            # Calcular direção de movimento (real - baseada em posições reais)
            direcao_lat = target_lat - current_lat
            direcao_lon = target_lon - current_lon

            # Features históricas (posições anteriores reais)
            prev_rows = tubaroes_df[
                (tubaroes_df["id_tubarao"] == row["id_tubarao"])
                & (tubaroes_df["tempo"] < current_time)
            ]

            lat_anterior = current_lat  # default
            lon_anterior = current_lon  # default

            if len(prev_rows) > 0:
                last_prev = prev_rows.iloc[-1]
                lat_anterior = last_prev["lat"]  # REAL: posição anterior
                lon_anterior = last_prev["lon"]  # REAL: posição anterior

            # Criar registro para ML (APENAS dados reais)
            ml_record = {
                # Features de entrada (X) - APENAS DADOS REAIS
                "lat_atual": current_lat,  # Posição atual real
                "lon_atual": current_lon,  # Posição atual real
                "hora_dia": hora_dia,  # Horário real
                "ssha_ambiente": ssha_ambiente,  # Dados SWOT reais (satélite)
                "chlor_a_ambiente": chlor_a_ambiente,  # Dados MODIS reais (satélite)
                "gradiente_ssha": gradiente_ssha,  # Variação espacial real SWOT
                "gradiente_chlor_a": gradiente_chlor_a,  # Variação espacial real MODIS
                "lat_anterior": lat_anterior,  # Posição anterior real
                "lon_anterior": lon_anterior,  # Posição anterior real
                # Targets de saída (y) - APENAS DADOS REAIS
                "target_lat": target_lat,  # Próxima posição real
                "target_lon": target_lon,  # Próxima posição real
                "distancia_percorrida": distancia_percorrida,  # Distância real calculada
                "direcao_lat": direcao_lat,  # Direção real calculada
                "direcao_lon": direcao_lon,  # Direção real calculada
                # Metadados
                "id_tubarao": row["id_tubarao"],  # ID do tubarão
                "tempo_atual": current_time,  # Tempo atual real
                "tempo_target": next_row["tempo"],  # Tempo target real
            }

            ml_data.append(ml_record)

    print(f"Features criadas usando APENAS dados reais:")
    print(f"  - Posições dos tubarões: lat/lon/tempo")
    print(f"  - Dados ambientais: SWOT SSHA + MODIS clorofila")
    print(f"  - Gradientes espaciais: variação do ambiente real")
    print(f"  - Targets: próxima posição real dos tubarões")
    print(f"  - NÃO inclui: velocidade, fadiga_nutricional, comportamento (simulados)")

    return pd.DataFrame(ml_data)


def build_kdtree_for_swot_files(
    swot_files: List[Dict],
) -> Tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói árvore KD para arquivos SWOT de uma data específica

    Args:
        swot_files: Lista de metadados de arquivos SWOT

    Returns:
        Tuple com (kdtree, lats, lons, ssha_values)
    """
    all_lats = []
    all_lons = []
    all_ssha = []

    print(f"Construindo árvore KD para {len(swot_files)} arquivos SWOT...")

    for swot_meta in tqdm(swot_files, desc="Processando SWOT"):
        try:
            ds = xr.open_dataset(swot_meta["file_path"])

            # Verificar se tem as variáveis corretas do SWOT
            if "ssha_karin" in ds.data_vars:
                # Extrair coordenadas do SWOT
                if (
                    "latitude_avg_ssh" in ds.data_vars
                    and "longitude_avg_ssh" in ds.data_vars
                ):
                    # Usar coordenadas médias se disponíveis
                    lats = ds["latitude_avg_ssh"].values
                    lons = ds["longitude_avg_ssh"].values
                else:
                    # Tentar extrair coordenadas de outras formas
                    print(
                        f"AVISO: Coordenadas não encontradas em {swot_meta['file_path']}"
                    )
                    continue

                ssha = ds["ssha_karin"].values

                # Garantir que as dimensões sejam compatíveis
                if lats.shape != ssha.shape:
                    # Se as dimensões não coincidem, tentar expandir coordenadas
                    if len(lats.shape) == 1 and len(ssha.shape) == 2:
                        # Expandir coordenadas 1D para 2D
                        lats_expanded = np.tile(lats[:, np.newaxis], (1, ssha.shape[1]))
                        lons_expanded = np.tile(lons[:, np.newaxis], (1, ssha.shape[1]))
                        lats = lats_expanded.flatten()
                        lons = lons_expanded.flatten()
                        ssha = ssha.flatten()
                    else:
                        print(
                            f"AVISO: Dimensões incompatíveis em {swot_meta['file_path']}"
                        )
                        continue

                # Filtrar valores válidos
                valid_mask = ~(np.isnan(lats) | np.isnan(lons) | np.isnan(ssha))

                if np.any(valid_mask):
                    all_lats.extend(lats[valid_mask])
                    all_lons.extend(lons[valid_mask])
                    all_ssha.extend(ssha[valid_mask])
                    print(
                        f"Adicionados {np.sum(valid_mask)} pontos válidos de {swot_meta['file_path']}"
                    )

            ds.close()

        except Exception as e:
            print(f"ERRO ao processar {swot_meta['file_path']}: {e}")
            continue

    if not all_lats:
        print("AVISO: Nenhum dado SWOT válido encontrado")
        return None, np.array([]), np.array([]), np.array([])

    # Construir KD-Tree
    coords = np.column_stack([all_lats, all_lons])
    kdtree = cKDTree(coords)

    print(f"Árvore KD SWOT criada: {len(all_lats)} pontos")
    return kdtree, np.array(all_lats), np.array(all_lons), np.array(all_ssha)


def extract_metadata(ncfile: str) -> Dict:
    """
    Extrai metadados de um arquivo NetCDF

    Args:
        ncfile: Caminho do arquivo NetCDF

    Returns:
        Dicionário com metadados do arquivo
    """
    try:
        ds = xr.open_dataset(ncfile)

        metadata = {
            "file_path": ncfile,
            "file_name": os.path.basename(ncfile),
            "file_size_mb": os.path.getsize(ncfile) / (1024 * 1024),
            "variables": list(ds.data_vars.keys()),
            "dimensions": dict(ds.dims),
            "created": datetime.now().isoformat(),
        }

        # Extrair informações específicas baseadas no tipo de arquivo
        if "ssha_karin" in ds.data_vars:
            # Arquivo SWOT
            metadata.update(
                {
                    "type": "SWOT",
                    "lat_min": float(ds.latitude.min()),
                    "lat_max": float(ds.latitude.max()),
                    "lon_min": float(ds.longitude.min()),
                    "lon_max": float(ds.longitude.max()),
                    "time_start": str(ds.time.min().values),
                    "time_end": str(ds.time.max().values),
                    "valid_points": int(np.sum(~np.isnan(ds.ssha_karin.values))),
                }
            )
        elif any(var in ds.data_vars for var in ["chlor_a", "BinList", "chlorophyll"]):
            # Arquivo MODIS
            try:
                # Tentar abrir grupo L3b
                ds_modis = xr.open_dataset(ncfile, group="level-3_binned_data")

                if "BinList" in ds_modis.data_vars:
                    binlist = ds_modis["BinList"]
                    binlist_values = binlist.values

                    if (
                        hasattr(binlist_values, "dtype")
                        and "bin_num" in binlist_values.dtype.names
                    ):
                        bin_nums = binlist_values["bin_num"]
                        lats, lons = converter_bin_para_lat_lon(bin_nums)

                        metadata.update(
                            {
                                "type": "MODIS",
                                "lat_min": float(np.nanmin(lats)),
                                "lat_max": float(np.nanmax(lats)),
                                "lon_min": float(np.nanmin(lons)),
                                "lon_max": float(np.nanmax(lons)),
                                "valid_points": len(bin_nums),
                                "date": extract_date_from_filename(ncfile),
                            }
                        )
            except:
                # Fallback para estrutura padrão
                metadata.update({"type": "MODIS", "valid_points": 0})
        else:
            # Verificar se é arquivo MODIS pelo nome
            if "MODIS" in os.path.basename(ncfile).upper():
                metadata.update(
                    {
                        "type": "MODIS",
                        "date": extract_date_from_filename(ncfile),
                        "valid_points": 0,
                    }
                )
            else:
                metadata["type"] = "UNKNOWN"

        ds.close()
        return metadata

    except Exception as e:
        print(f"ERRO ao extrair metadados de {ncfile}: {e}")
        return {
            "file_path": ncfile,
            "file_name": os.path.basename(ncfile),
            "type": "ERROR",
            "error": str(e),
            "created": datetime.now().isoformat(),
        }


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
            import re

            match = re.search(r"(\d{8})", basename)
            if match:
                return match.group(1)  # 20240101

        return "unknown"
    except:
        return "unknown"


def converter_bin_para_lat_lon(bin_nums):
    """Converte números de bin MODIS para coordenadas lat/lon"""
    num_rows = 2160
    num_cols = 4320

    lats = []
    lons = []

    for bin_num in bin_nums:
        row = bin_num // num_cols
        col = bin_num % num_cols

        lat = 90.0 - (row + 0.5) * (180.0 / num_rows)
        lon = (col + 0.5) * (360.0 / num_cols) - 180.0

        lats.append(lat)
        lons.append(lon)

    return np.array(lats), np.array(lons)


def cache_metadata(files: List[str], cachefile: str) -> List[Dict]:
    """
    Cache metadados de arquivos para evitar reprocessamento

    Args:
        files: Lista de arquivos para processar
        cachefile: Caminho do arquivo de cache

    Returns:
        Lista de metadados
    """
    # Verificar se cache existe e é recente
    if os.path.exists(cachefile):
        try:
            with open(cachefile, "rb") as f:
                cached_metadata = pickle.load(f)

            # Verificar se todos os arquivos ainda existem
            existing_files = [meta["file_path"] for meta in cached_metadata]
            missing_files = [f for f in files if f not in existing_files]

            if not missing_files:
                print(f"Cache encontrado: {len(cached_metadata)} arquivos")
                return cached_metadata
            else:
                print(f"Cache parcial: {len(missing_files)} novos arquivos encontrados")
        except:
            print("Cache corrompido, recriando...")

    # Processar metadados
    print(f"Extraindo metadados de {len(files)} arquivos...")
    metadata_list = []

    for file_path in tqdm(files, desc="Extraindo metadados"):
        metadata = extract_metadata(file_path)
        metadata_list.append(metadata)

    # Salvar cache
    with open(cachefile, "wb") as f:
        pickle.dump(metadata_list, f)

    print(f"Metadados salvos em cache: {cachefile}")
    return metadata_list


def build_temporal_index(metadata_list: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Constrói índice temporal dos arquivos

    Args:
        metadata_list: Lista de metadados

    Returns:
        Dicionário com arquivos agrupados por data
    """
    temporal_index = {}

    for metadata in metadata_list:
        if metadata["type"] == "MODIS":
            date = metadata.get("date", "unknown")
            if date not in temporal_index:
                temporal_index[date] = {"swot": [], "modis": []}
            temporal_index[date]["modis"].append(metadata)
        elif metadata["type"] == "SWOT":
            # Extrair data do nome do arquivo (não do tempo interno)
            date = extract_date_from_filename(metadata["file_path"])
            if date != "unknown":
                if date not in temporal_index:
                    temporal_index[date] = {"swot": [], "modis": []}
                temporal_index[date]["swot"].append(metadata)

    print(f"Índice temporal criado: {len(temporal_index)} datas únicas")
    return temporal_index


def build_kdtree_for_modis_files(
    modis_files: List[Dict],
) -> Tuple[cKDTree, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constrói árvore KD para arquivos MODIS de uma data específica

    Args:
        modis_files: Lista de metadados de arquivos MODIS

    Returns:
        Tuple com (kdtree, lats, lons, chlor_values)
    """
    all_lats = []
    all_lons = []
    all_chlor = []

    print(f"Construindo árvore KD para {len(modis_files)} arquivos MODIS...")

    for modis_meta in tqdm(modis_files, desc="Processando MODIS"):
        try:
            ds = xr.open_dataset(modis_meta["file_path"], group="level-3_binned_data")

            if "BinList" in ds.data_vars and "chlor_a" in ds.data_vars:
                binlist = ds["BinList"]
                binlist_values = binlist.values

                if (
                    hasattr(binlist_values, "dtype")
                    and "bin_num" in binlist_values.dtype.names
                ):
                    bin_nums = binlist_values["bin_num"]
                    lats, lons = converter_bin_para_lat_lon(bin_nums)

                    # Processar clorofila
                    chlor_values = ds["chlor_a"].values
                    if (
                        hasattr(chlor_values, "dtype")
                        and "sum" in chlor_values.dtype.names
                    ):
                        chlor_sum = chlor_values["sum"]
                        nobs = binlist_values["nobs"]
                        chlor_media = np.where(nobs > 0, chlor_sum / nobs, np.nan)

                        # Filtrar pontos válidos
                        mask_validos = ~np.isnan(chlor_media) & (chlor_media > 0)

                        if np.any(mask_validos):
                            all_lats.extend(lats[mask_validos])
                            all_lons.extend(lons[mask_validos])
                            all_chlor.extend(chlor_media[mask_validos])

            ds.close()

        except Exception as e:
            print(f"ERRO ao processar {modis_meta['file_name']}: {e}")
            continue

    if not all_lats:
        return None, np.array([]), np.array([]), np.array([])

    # Construir árvore KD
    coords = np.column_stack([all_lats, all_lons])
    kdtree = cKDTree(coords)

    print(f"Árvore KD criada: {len(all_lats)} pontos MODIS")
    return kdtree, np.array(all_lats), np.array(all_lons), np.array(all_chlor)


def process_swot_file(
    swot_file: str,
    modis_kdtree: cKDTree,
    modis_lats: np.ndarray,
    modis_lons: np.ndarray,
    modis_chlor: np.ndarray,
    out_csv: str,
    tolerance_deg: float = 1.0,
    batch_size: int = 10000,
) -> int:
    """
    Processa um arquivo SWOT contra uma árvore KD MODIS

    Args:
        swot_file: Caminho do arquivo SWOT
        modis_kdtree: Árvore KD dos pontos MODIS
        modis_lats: Latitudes MODIS
        modis_lons: Longitudes MODIS
        modis_chlor: Valores de clorofila MODIS
        out_csv: Arquivo CSV de saída
        tolerance_deg: Tolerância em graus
        batch_size: Tamanho do lote

    Returns:
        Número de pontos de interseção encontrados
    """
    if modis_kdtree is None:
        return 0

    try:
        ds = xr.open_dataset(swot_file)

        if "ssha_karin" not in ds.data_vars:
            ds.close()
            return 0

        # Extrair dados SWOT
        ssha_data = ds["ssha_karin"].values
        mask_validos = ~np.isnan(ssha_data)

        if not np.any(mask_validos):
            ds.close()
            return 0

        swot_lats = ds["latitude"].values[mask_validos]
        swot_lons = ds["longitude"].values[mask_validos]
        swot_ssha = ssha_data[mask_validos]

        # Processar tempo
        time_data = ds["time"].values
        if len(time_data.shape) == 1:
            time_expanded = np.repeat(
                time_data[:, np.newaxis], ssha_data.shape[1], axis=1
            )
            swot_time = time_expanded[mask_validos]
        else:
            swot_time = time_data[mask_validos]

        ds.close()

        # Buscar interseções espaciais
        swot_coords = np.column_stack([swot_lats, swot_lons])
        pontos_interseccao = []

        # Processar em lotes
        for i in range(0, len(swot_coords), batch_size):
            end_idx = min(i + batch_size, len(swot_coords))
            swot_batch = swot_coords[i:end_idx]

            # Buscar pontos MODIS próximos
            distances, indices = modis_kdtree.query(
                swot_batch, k=1, distance_upper_bound=tolerance_deg
            )

            # Filtrar pontos dentro da tolerância
            valid_mask = distances < tolerance_deg

            for j, (is_valid, dist, modis_idx) in enumerate(
                zip(valid_mask, distances, indices)
            ):
                if is_valid:
                    swot_idx = i + j
                    ponto = {
                        "swot_file": os.path.basename(swot_file),
                        "modis_file": "multiple",  # Pode ser de vários arquivos
                        "lat": swot_lats[swot_idx],
                        "lon": swot_lons[swot_idx],
                        "ssha": swot_ssha[swot_idx],
                        "chlor_a": modis_chlor[modis_idx],
                        "distancia": dist,
                        "time": swot_time[swot_idx],
                    }
                    pontos_interseccao.append(ponto)

        # Salvar resultados incrementais
        if pontos_interseccao:
            df = pd.DataFrame(pontos_interseccao)

            # Adicionar metadados
            df["correlation"] = np.nan  # Será calculado depois
            df["data_type"] = "real_intersection_scalable"
            df["source"] = "SWOT_MODIS_NASA_SCALABLE"
            df["date_processed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df["tolerance_degrees"] = tolerance_deg
            df["algorithm"] = "cKDTree_scalable_search"

            # Salvar com append
            file_exists = os.path.exists(out_csv)
            df.to_csv(out_csv, mode="a", header=not file_exists, index=False)

        return len(pontos_interseccao)

    except Exception as e:
        print(f"ERRO ao processar SWOT {swot_file}: {e}")
        return 0


def process_temporal_window(
    date: str,
    swot_files: List[Dict],
    modis_files: List[Dict],
    out_csv: str,
    tolerance_deg: float = 1.0,
) -> int:
    """
    Processa uma janela temporal (um dia)

    Args:
        date: Data no formato YYYYMMDD
        swot_files: Lista de arquivos SWOT
        modis_files: Lista de arquivos MODIS
        out_csv: Arquivo CSV de saída
        tolerance_deg: Tolerância em graus

    Returns:
        Número total de pontos de interseção
    """
    if not swot_files or not modis_files:
        return 0

    print(f"\n=== PROCESSANDO JANELA TEMPORAL: {date} ===")
    print(f"SWOT: {len(swot_files)} arquivos")
    print(f"MODIS: {len(modis_files)} arquivos")

    # Verificar cache da árvore KD
    cache_file = os.path.join(CACHE_DIR, f"modis_tree_{date}.pkl")

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                kdtree_data = pickle.load(f)
            modis_kdtree, modis_lats, modis_lons, modis_chlor = kdtree_data
            print(f"Árvore KD carregada do cache: {len(modis_lats)} pontos")
        except:
            print("Cache corrompido, recriando...")
            modis_kdtree, modis_lats, modis_lons, modis_chlor = (
                build_kdtree_for_modis_files(modis_files)
            )

            # Salvar cache
            if modis_kdtree is not None:
                with open(cache_file, "wb") as f:
                    pickle.dump((modis_kdtree, modis_lats, modis_lons, modis_chlor), f)
    else:
        modis_kdtree, modis_lats, modis_lons, modis_chlor = (
            build_kdtree_for_modis_files(modis_files)
        )

        # Salvar cache
        if modis_kdtree is not None:
            with open(cache_file, "wb") as f:
                pickle.dump((modis_kdtree, modis_lats, modis_lons, modis_chlor), f)

    if modis_kdtree is None:
        print(f"AVISO: Nenhum dado MODIS válido para {date}")
        return 0

    # Processar arquivos SWOT
    total_intersections = 0

    if N_JOBS == 1:
        # Processamento sequencial
        for swot_meta in tqdm(swot_files, desc=f"Processando SWOT {date}"):
            intersections = process_swot_file(
                swot_meta["file_path"],
                modis_kdtree,
                modis_lats,
                modis_lons,
                modis_chlor,
                out_csv,
                tolerance_deg,
                BATCH_SIZE,
            )
            total_intersections += intersections
    else:
        # Processamento paralelo
        def process_swot_wrapper(swot_meta):
            return process_swot_file(
                swot_meta["file_path"],
                modis_kdtree,
                modis_lats,
                modis_lons,
                modis_chlor,
                out_csv,
                tolerance_deg,
                BATCH_SIZE,
            )

        results = joblib.Parallel(n_jobs=N_JOBS, backend="threading")(
            joblib.delayed(process_swot_wrapper)(swot_meta) for swot_meta in swot_files
        )

        total_intersections = sum(results)

    print(f"Pontos de interseção encontrados: {total_intersections}")
    return total_intersections


def calculate_final_correlation(csv_file: str) -> float:
    """
    Calcula correlação final entre SSHA e clorofila

    Args:
        csv_file: Arquivo CSV com os dados

    Returns:
        Correlação de Pearson
    """
    try:
        df = pd.read_csv(csv_file)

        if len(df) < 10:
            return 0.0

        correlation = np.corrcoef(df["ssha"], df["chlor_a"])[0, 1]

        # Atualizar correlação no CSV
        df["correlation"] = correlation
        df.to_csv(csv_file, index=False)

        return correlation

    except Exception as e:
        print(f"ERRO ao calcular correlação: {e}")
        return 0.0


def generate_final_report(csv_file: str, start_time: datetime) -> None:
    """
    Gera relatório final da análise

    Args:
        csv_file: Arquivo CSV com os dados
        start_time: Tempo de início da execução
    """
    try:
        # Tentar ler CSV com tratamento de erro
        try:
            df = pd.read_csv(csv_file)
        except pd.errors.ParserError as e:
            print(f"ERRO ao ler CSV: {e}")
            print("Tentando corrigir arquivo CSV...")

            # Ler linha por linha e corrigir
            with open(csv_file, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Verificar header
            header = lines[0].strip().split(",")
            expected_fields = len(header)
            print(f"Campos esperados: {expected_fields}")

            # Filtrar linhas com número correto de campos
            clean_lines = []
            for i, line in enumerate(lines):
                fields = line.strip().split(",")
                if len(fields) == expected_fields:
                    clean_lines.append(line)
                else:
                    print(
                        f"Removendo linha {i+1} com {len(fields)} campos (esperado: {expected_fields})"
                    )

            # Recriar CSV limpo
            temp_file = csv_file + ".temp"
            with open(temp_file, "w", encoding="utf-8") as f:
                f.writelines(clean_lines)

            # Substituir arquivo original
            os.replace(temp_file, csv_file)

            df = pd.read_csv(csv_file)

        report_file = os.path.join(DATA_DIR, "relatorio_treinamento_ia.txt")

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("RELATÓRIO DE COLETA DE DADOS SWOT x MODIS\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Data de coleta: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tempo total: {datetime.now() - start_time}\n\n")

            f.write("CONFIGURAÇÕES DE COLETA:\n")
            f.write(f"- Tolerância espacial: {TOLERANCE_DEGREES}°\n")
            f.write(f"- Tamanho do lote: {BATCH_SIZE:,}\n")
            f.write(f"- Paralelização: {N_JOBS} cores\n\n")

            f.write("DADOS COLETADOS:\n")
            f.write(f"- Total de amostras: {len(df):,}\n")

            # Calcular correlação se possível
            try:
                correlacao = df["ssha"].corr(df["chlor_a"])
                f.write(f"- Correlação SSHA vs Clorofila: {correlacao:.6f}\n")
            except:
                f.write("- Correlação: Não calculável\n")

            f.write(
                f"- Range SSHA: {df['ssha'].min():.3f} a {df['ssha'].max():.3f} m\n"
            )
            f.write(
                f"- Range Clorofila: {df['chlor_a'].min():.3f} a {df['chlor_a'].max():.3f} mg/m³\n"
            )
            f.write(f"- Distância média: {df['distancia'].mean():.3f}°\n")
            f.write(f"- Arquivos SWOT únicos: {df['swot_file'].nunique()}\n")
            f.write(f"- Arquivos MODIS únicos: {df['modis_file'].nunique()}\n\n")

            f.write("COBERTURA GEOGRÁFICA:\n")
            f.write(f"- Latitude: {df['lat'].min():.3f} a {df['lat'].max():.3f}°\n")
            f.write(f"- Longitude: {df['lon'].min():.3f} a {df['lon'].max():.3f}°\n\n")

            f.write("QUALIDADE DOS DADOS:\n")
            f.write(
                f"- SSHA válidos: {df['ssha'].notna().sum():,} ({df['ssha'].notna().mean()*100:.1f}%)\n"
            )
            f.write(
                f"- Clorofila válidos: {df['chlor_a'].notna().sum():,} ({df['chlor_a'].notna().mean()*100:.1f}%)\n"
            )
            f.write(
                f"- Coordenadas válidas: {df['lat'].notna().sum():,} ({df['lat'].notna().mean()*100:.1f}%)\n\n"
            )

            f.write("DADOS PRONTOS PARA:\n")
            f.write("- Treinamento de IA\n")
            f.write("- Análise espacial\n")
            f.write("- Modelagem de rastreamento\n\n")

            f.write("ARQUIVOS GERADOS:\n")
            f.write(f"- Dados: {csv_file}\n")
            f.write(f"- Relatório: {report_file}\n")

        print(f"\nRelatório final salvo: {report_file}")

    except Exception as e:
        print(f"ERRO ao gerar relatório: {e}")


def main():
    """Função principal do processamento escalável para IA de comportamento"""
    print("COLETA DE DADOS PARA IA DE COMPORTAMENTO ANIMAL")
    print("=" * 60)
    print("SWOT + MODIS + Dados de Tubaroes -> Dataset ML")
    print("=" * 60)

    start_time = datetime.now()

    # === ETAPA 1: Descoberta de arquivos ===
    print("\n1. DESCOBRINDO ARQUIVOS...")
    swot_files = discover_files(f"{DATA_DIR}/swot", "*.nc")
    modis_files = discover_files(f"{DATA_DIR}/modis", "*.nc")
    tubaroes_files = discover_tubaroes_files()

    if not swot_files and not modis_files:
        print("AVISO: Nenhum arquivo encontrado. Coloque arquivos .nc nas pastas:")
        print(f"  - {DATA_DIR}/swot/ (arquivos SWOT)")
        print(f"  - {DATA_DIR}/modis/ (arquivos MODIS)")
        return

    if not tubaroes_files:
        print("AVISO: Arquivos de tubarões não encontrados!")
        print("Continuando apenas com análise SWOT x MODIS...")
    else:
        print(f"Arquivos de tubarões encontrados: {len(tubaroes_files)}")

    # === ETAPA 2: Cache de metadados ===
    print("\n2. PROCESSANDO METADADOS...")

    swot_metadata = cache_metadata(swot_files, os.path.join(CACHE_DIR, "swot_meta.pkl"))
    modis_metadata = cache_metadata(
        modis_files, os.path.join(CACHE_DIR, "modis_meta.pkl")
    )

    # === ETAPA 3: Índice temporal ===
    print("\n3. CONSTRUINDO ÍNDICE TEMPORAL...")
    all_metadata = swot_metadata + modis_metadata
    temporal_index = build_temporal_index(all_metadata)

    if not temporal_index:
        print("ERRO: Nenhuma correspondência temporal encontrada")
        return

    # === ETAPA 4: Processamento por janela temporal ===
    print("\n4. PROCESSANDO JANELAS TEMPORAIS...")

    # Arquivos de saída
    out_csv_swot_modis = os.path.join(DATA_DIR, "dados_swot_modis.csv")
    out_csv_ml_training = os.path.join(
        DATA_DIR, "dados_treinamento_ia_comportamento.csv"
    )

    # Limpar arquivos de saída se existirem
    for out_file in [out_csv_swot_modis, out_csv_ml_training]:
        if os.path.exists(out_file):
            os.remove(out_file)

    total_intersections = 0
    total_tubaroes_analyzed = 0
    total_ml_records = 0
    all_ml_data = []

    for date in tqdm(sorted(temporal_index.keys()), desc="Processando datas"):
        swot_files_date = temporal_index[date]["swot"]
        modis_files_date = temporal_index[date]["modis"]

        # Processar dados SWOT-MODIS (dados ambientais)
        intersections = process_temporal_window(
            date,
            swot_files_date,
            modis_files_date,
            out_csv_swot_modis,
            TOLERANCE_DEGREES,
        )
        total_intersections += intersections

        # Processar dados dos tubarões para ML se disponíveis
        if tubaroes_files and date in tubaroes_files:
            tubaroes_df = load_tubaroes_data(date, tubaroes_files)
            if tubaroes_df is not None:
                total_tubaroes_analyzed += len(tubaroes_df)
                print(
                    f"  Tubarões analisados para {date}: {len(tubaroes_df)} registros"
                )

                # Construir KD-Tree para MODIS desta data
                modis_kdtree, modis_lats, modis_lons, modis_chlor = (
                    build_kdtree_for_modis_files(modis_files_date)
                )

                # Preparar dados MODIS no formato esperado
                modis_data = []
                for i in range(len(modis_lats)):
                    modis_data.append(
                        {
                            "lat": modis_lats[i],
                            "lon": modis_lons[i],
                            "chlor_a": modis_chlor[i],
                        }
                    )

                modis_kdtree_data = {"kdtree": modis_kdtree, "data": modis_data}

                # Construir KD-Tree para SWOT desta data
                swot_kdtree, swot_lats, swot_lons, swot_ssha = (
                    build_kdtree_for_swot_files(swot_files_date)
                )

                # Verificar se a árvore KD foi criada com sucesso
                if swot_kdtree is None:
                    print(f"  AVISO: Não foi possível criar árvore KD SWOT para {date}")
                    swot_kdtree_data = {"kdtree": None, "data": []}
                else:
                    # Preparar dados SWOT no formato esperado
                    swot_data = []
                    for i in range(len(swot_lats)):
                        swot_data.append(
                            {
                                "lat": swot_lats[i],
                                "lon": swot_lons[i],
                                "ssha": swot_ssha[i],
                            }
                        )

                    swot_kdtree_data = {"kdtree": swot_kdtree, "data": swot_data}

                # Criar features e targets para ML (apenas dados reais)
                try:
                    ml_data_date = create_ml_features_targets_reais(
                        tubaroes_df, modis_kdtree_data, swot_kdtree_data
                    )
                except Exception as e:
                    print(f"  ERRO ao criar features ML: {e}")
                    ml_data_date = pd.DataFrame()

                if len(ml_data_date) > 0:
                    all_ml_data.append(ml_data_date)
                    total_ml_records += len(ml_data_date)
                    print(f"  Registros ML criados para {date}: {len(ml_data_date)}")

    # Salvar dados de treinamento ML
    if all_ml_data:
        print(f"\nSalvando {total_ml_records} registros de treinamento ML...")
        combined_ml_data = pd.concat(all_ml_data, ignore_index=True)
        combined_ml_data.to_csv(out_csv_ml_training, index=False, encoding="utf-8")
        print(f"Dataset de treinamento salvo: {out_csv_ml_training}")

    # === ETAPA 5: Análise final ===
    print("\n5. ANÁLISE FINAL...")

    correlation_swot_modis = 0.0
    if os.path.exists(out_csv_swot_modis):
        correlation_swot_modis = calculate_final_correlation(out_csv_swot_modis)
        generate_final_report(out_csv_swot_modis, start_time)

    print(f"\nSUCESSO: COLETA DE DADOS CONCLUÍDA!")
    print(f"   Pontos SWOT-MODIS coletados: {total_intersections:,}")
    print(f"   Registros de tubarões analisados: {total_tubaroes_analyzed:,}")
    print(f"   Registros ML para treinamento: {total_ml_records:,}")
    print(f"   Correlação SWOT-MODIS: {correlation_swot_modis:.6f}")
    print(f"   Arquivo dados ambientais: {out_csv_swot_modis}")
    print(f"   Arquivo treinamento IA: {out_csv_ml_training}")
    print(f"   Tempo total: {datetime.now() - start_time}")
    print(f"\n[OK] Dados ambientais SWOT-MODIS coletados")
    print(f"[OK] Dataset de treinamento IA de comportamento criado")
    print(f"[OK] Matching automático por data implementado")
    print(f"[OK] Features e targets ML gerados")


if __name__ == "__main__":
    main()
