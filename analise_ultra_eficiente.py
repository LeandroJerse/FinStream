#!/usr/bin/env python3
"""
Análise Ultra-Eficiente e Escalável - SWOT x MODIS
NASA Ocean Data Coherence Checker - FinStream Project
Processamento automático de múltiplos arquivos com caching inteligente
"""

import glob
import os
import pickle
import warnings
from datetime import datetime, timedelta
from pathlib import Path
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

# Criar diretórios necessários
for dir_name in [
    CACHE_DIR,
    RESULTS_DIR,
    DATA_DIR,
    f"{DATA_DIR}/swot",
    f"{DATA_DIR}/modis",
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
    """Função principal do processamento escalável"""
    print("COLETA DE DADOS SWOT x MODIS PARA TREINAMENTO DE IA")
    print("=" * 60)
    print("Matching automático por data - Apenas coleta de dados")
    print("=" * 60)

    start_time = datetime.now()

    # === ETAPA 1: Descoberta de arquivos ===
    print("\n1. DESCOBRINDO ARQUIVOS...")
    swot_files = discover_files(f"{DATA_DIR}/swot", "*.nc")
    modis_files = discover_files(f"{DATA_DIR}/modis", "*.nc")

    if not swot_files and not modis_files:
        print("AVISO: Nenhum arquivo encontrado. Coloque arquivos .nc nas pastas:")
        print(f"  - {DATA_DIR}/swot/ (arquivos SWOT)")
        print(f"  - {DATA_DIR}/modis/ (arquivos MODIS)")
        return

    # === ETAPA 2: Cache de metadados ===
    print("\n2. PROCESSANDO METADADOS...")
    all_files = swot_files + modis_files

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

    out_csv = os.path.join(DATA_DIR, "dados_treinamento_ia.csv")

    # Limpar arquivo de saída se existir
    if os.path.exists(out_csv):
        os.remove(out_csv)

    total_intersections = 0

    for date in tqdm(sorted(temporal_index.keys()), desc="Processando datas"):
        swot_files_date = temporal_index[date]["swot"]
        modis_files_date = temporal_index[date]["modis"]

        intersections = process_temporal_window(
            date, swot_files_date, modis_files_date, out_csv, TOLERANCE_DEGREES
        )
        total_intersections += intersections

    # === ETAPA 5: Análise final ===
    print("\n5. ANÁLISE FINAL...")

    if os.path.exists(out_csv):
        correlation = calculate_final_correlation(out_csv)
        generate_final_report(out_csv, start_time)

        print(f"\nSUCESSO: COLETA DE DADOS CONCLUÍDA!")
        print(f"   Pontos coletados: {total_intersections:,}")
        print(f"   Correlação: {correlation:.6f}")
        print(f"   Arquivo de dados: {out_csv}")
        print(f"   Tempo total: {datetime.now() - start_time}")
        print(f"\n[OK] Dados coletados e salvos para treinamento de IA")
        print(f"[OK] Matching automático por data implementado")
        print(f"[OK] Coleta de dados SWOT-MODIS concluída")
    else:
        print("ERRO: Nenhum dado processado")


if __name__ == "__main__":
    main()
