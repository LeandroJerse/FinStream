#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gerador de Dados de Validação para IA - Tubarões
===============================================

Gera dados sintéticos de tubarões para validação da IA treinada.
Baseado no simular_tubaroes.py mas SEM as colunas p_forrageio e comportamento.

Autor: Sistema de Análise Oceanográfica Avançada
Data: 2024-01-01
"""

import glob
import os
import re
import struct
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree
from scipy.special import expit as sigmoid
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES AVANÇADAS
# =============================================================================

# Parâmetros da simulação
N_TUBAROES = 1
PINGS_POR_TUBARAO = (
    288  # 24 horas * 12 pings/hora (5 min intervalo) = 288 pings por dia
)
INTERVALO_PING_MINUTOS = 5
DATA_INICIO = "2024-01-01 00:00:00"
DATA_FIM = "2024-01-24 23:55:00"  # Última data disponível nos dados MODIS/SWOT

# Parâmetros de telemetria do dispositivo
# Baseado em especificações reais de tags de rastreamento
PROFUNDIDADE_MAXIMA_M = 1000  # Profundidade máxima típica para tubarões
TEMPERATURA_OCEANO_SURFACE = 25.0  # °C
TEMPERATURA_OCEANO_DEEP = 4.0  # °C
BATERIA_INICIAL_MV = 3700  # 3.7V em mV
BATERIA_MINIMA_MV = 3000  # 3.0V mínimo

# Parâmetros de aceleração (em mg)
ACC_RANGE_MG = 16000  # ±16g = ±16000mg
ACC_NOISE_MG = 50  # Ruído típico do sensor

# Parâmetros de giroscópio (em mdps)
GYR_RANGE_MDPS = 2000000  # ±2000°/s = ±2000000mdps
GYR_NOISE_MDPS = 1000  # Ruído típico do sensor

# Diretórios
DADOS_AMBIENTAIS = "data/dados_unificados_final.csv"
OUTPUT_DIR = "data/IA_TREINADA"

# =============================================================================
# FUNÇÕES DE TELEMETRIA
# =============================================================================


def calcular_crc16_ccitt(data):
    """
    Calcula CRC-16/CCITT para verificação de integridade dos dados.

    Args:
        data: bytes dos dados de telemetria

    Returns:
        int: CRC-16 calculado
    """
    crc = 0xFFFF
    polynomial = 0x1021

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ polynomial
            else:
                crc <<= 1
            crc &= 0xFFFF

    return crc


def converter_coordenadas_para_telemetria(lat, lon):
    """
    Converte coordenadas float para formato de telemetria (graus * 1e-4).

    Args:
        lat: latitude em graus decimais
        lon: longitude em graus decimais

    Returns:
        tuple: (lat_int, lon_int) em formato de telemetria
    """
    lat_int = int(lat * 1e4)
    lon_int = int(lon * 1e4)

    # Limitar a 3 bytes com sinal (-8,388,608 a 8,388,607)
    lat_int = max(-8388608, min(8388607, lat_int))
    lon_int = max(-8388608, min(8388607, lon_int))

    return lat_int, lon_int


def simular_dados_sensores(profundidade_m, tempo_dia, nivel_fome):
    """
    Simula dados dos sensores baseado no contexto ambiental.

    Args:
        profundidade_m: profundidade atual em metros
        tempo_dia: hora do dia (0-24)
        nivel_fome: nível de fome (0-1)

    Returns:
        dict: dados dos sensores simulados
    """
    # Profundidade em decímetros
    depth_dm = int(profundidade_m * 10)

    # Temperatura baseada na profundidade (termoclina)
    if profundidade_m < 50:
        temp_c = TEMPERATURA_OCEANO_SURFACE - (profundidade_m * 0.1)
    else:
        temp_c = TEMPERATURA_OCEANO_SURFACE - 5.0 - ((profundidade_m - 50) * 0.02)
        temp_c = max(TEMPERATURA_OCEANO_DEEP, temp_c)

    # Adicionar variação temporal (ritmo circadiano)
    temp_c += 2.0 * np.sin(2 * np.pi * tempo_dia / 24)
    temp_cC = int(temp_c * 100)  # Converter para centésimos de grau

    # Bateria (degradação ao longo do tempo)
    bateria_mv = BATERIA_INICIAL_MV - (nivel_fome * 200)
    bateria_mv = max(BATERIA_MINIMA_MV, bateria_mv)
    batt_mV = int(bateria_mv)

    # Acelerômetro (baseado na atividade)
    atividade = 0.5 + 0.3 * np.sin(2 * np.pi * tempo_dia / 24) + nivel_fome * 0.2

    # Aceleração em mg (limitada para evitar overflow)
    acc_x = int(
        np.clip(np.random.normal(0, ACC_NOISE_MG) + atividade * 1000, -16000, 16000)
    )
    acc_y = int(
        np.clip(np.random.normal(0, ACC_NOISE_MG) + atividade * 800, -16000, 16000)
    )
    acc_z = int(
        np.clip(np.random.normal(0, ACC_NOISE_MG) + atividade * 1200, -16000, 16000)
    )

    # Giroscópio (orientação e rotação) - limitado para evitar overflow
    gyr_x = int(np.clip(np.random.normal(0, GYR_NOISE_MDPS), -2000000, 2000000))
    gyr_y = int(np.clip(np.random.normal(0, GYR_NOISE_MDPS), -2000000, 2000000))
    gyr_z = int(np.clip(np.random.normal(0, GYR_NOISE_MDPS), -2000000, 2000000))

    return {
        "depth_dm": depth_dm,
        "temp_cC": temp_cC,
        "batt_mV": batt_mV,
        "acc_x": acc_x,
        "acc_y": acc_y,
        "acc_z": acc_z,
        "gyr_x": gyr_x,
        "gyr_y": gyr_y,
        "gyr_z": gyr_z,
    }


def calcular_profundidade_comportamental(tempo_dia, ssha, chlor_a):
    """
    Calcula profundidade baseada em fatores ambientais e temporais.

    Args:
        tempo_dia: hora do dia (0-24)
        ssha: Sea Surface Height Anomaly
        chlor_a: concentração de clorofila-a

    Returns:
        float: profundidade em metros
    """
    # Profundidade base
    profundidade_base = 20.0

    # Variação circadiana (mais ativo durante dia)
    if 6 <= tempo_dia <= 18:  # Dia
        profundidade_base += 10.0
    else:  # Noite
        profundidade_base += 50.0

    # Influência do SSHA (redemoinhos)
    if ssha > 0.1:  # Redemoinho anticiclônico
        profundidade_base += 100.0  # Mergulhos mais profundos
    elif ssha < -0.1:  # Redemoinho ciclônico
        profundidade_base += 50.0

    # Influência da clorofila (produtividade)
    if chlor_a > 0.5:  # Alta produtividade
        profundidade_base += 30.0  # Busca por presas em profundidade

    # Adicionar variação aleatória
    profundidade_base += np.random.normal(0, 20.0)

    # Limitar profundidade
    return max(5.0, min(PROFUNDIDADE_MAXIMA_M, profundidade_base))


def converter_bin_para_lat_lon(bin_nums):
    """
    Converte números de bin MODIS para coordenadas lat/lon.

    Args:
        bin_nums: array de números de bin

    Returns:
        tuple: (lats, lons) arrays de coordenadas
    """
    # Conversão de bin para lat/lon baseada na especificação MODIS
    # Cada bin representa uma área específica no oceano
    lats = []
    lons = []

    for bin_num in bin_nums:
        # Conversão simplificada - em implementação real seria mais complexa
        # Para fins de simulação, usar uma conversão aproximada
        lat = (bin_num // 3600) * 0.1 - 90.0
        lon = (bin_num % 3600) * 0.1 - 180.0
        lats.append(lat)
        lons.append(lon)

    return np.array(lats), np.array(lons)


def carregar_dados_swot(swot_files):
    """
    Carrega dados SWOT de múltiplos arquivos.

    Args:
        swot_files: lista de caminhos para arquivos SWOT

    Returns:
        tuple: (lats, lons, ssha) arrays
    """
    all_lats = []
    all_lons = []
    all_ssha = []

    for filepath in swot_files:
        try:
            ds = xr.open_dataset(filepath)

            # Extrair coordenadas e SSHA
            lats = ds["latitude"].values.flatten()
            lons = ds["longitude"].values.flatten()
            ssha = ds["ssha_karin"].values.flatten()

            # Converter longitude de 0-360 para -180 a 180
            lons = np.where(lons > 180, lons - 360, lons)

            # Filtrar valores válidos
            valid_mask = ~np.isnan(ssha)

            if np.any(valid_mask):
                all_lats.extend(lats[valid_mask])
                all_lons.extend(lons[valid_mask])
                all_ssha.extend(ssha[valid_mask])

            ds.close()

        except Exception as e:
            print(f"      AVISO: Erro ao ler SWOT {os.path.basename(filepath)}: {e}")
            continue

    return np.array(all_lats), np.array(all_lons), np.array(all_ssha)


def carregar_dados_modis(modis_files):
    """
    Carrega dados MODIS de múltiplos arquivos.

    Args:
        modis_files: lista de caminhos para arquivos MODIS

    Returns:
        tuple: (lats, lons, chlor_a) arrays
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


def gerar_dados_ambientais_sinteticos():
    """
    Gera dados ambientais sintéticos para validação.

    Returns:
        tuple: (df, kdtree, coords_array)
    """
    print("Gerando dados ambientais sintéticos...")

    # Gerar coordenadas oceânicas (área do Atlântico Sul)
    np.random.seed(42)
    n_pontos = 10000

    # Área oceânica: -40° a -20° lat, -50° a -30° lon
    lats = np.random.uniform(-40, -20, n_pontos)
    lons = np.random.uniform(-50, -30, n_pontos)

    # Gerar SSHA sintético (variação típica de redemoinhos)
    ssha = np.random.normal(0, 0.15, n_pontos)

    # Gerar clorofila sintética (valores típicos oceânicos)
    chlor_a = np.random.lognormal(mean=-2, sigma=1, size=n_pontos)
    chlor_a = np.clip(chlor_a, 0.01, 5.0)  # Limitar a valores realistas

    # Criar DataFrame
    df_sintetico = pd.DataFrame(
        {"lat": lats, "lon": lons, "ssha": ssha, "chlor_a": chlor_a}
    )

    print(f"Dados sintéticos gerados: {len(df_sintetico):,} pontos")
    print(
        f"SSHA range: {df_sintetico['ssha'].min():.2f} a {df_sintetico['ssha'].max():.2f}"
    )
    print(
        f"Chlor_a range: {df_sintetico['chlor_a'].min():.4f} a {df_sintetico['chlor_a'].max():.4f}"
    )

    # Criar array de coordenadas para KDTree
    coords = np.column_stack([df_sintetico["lat"], df_sintetico["lon"]])
    kdtree = cKDTree(coords)

    return df_sintetico, kdtree, coords


def carregar_dados_ambientais_por_data(data):
    """
    Carrega dados ambientais (SWOT + MODIS) para uma data específica.

    Args:
        data: Data no formato YYYY-MM-DD

    Returns:
        tuple: (swot_lats, swot_lons, swot_ssha, modis_lats, modis_lons, modis_chlor)
    """
    data_str = data.replace("-", "")

    # Encontrar arquivos SWOT para a data
    swot_files = []
    for root, dirs, files in os.walk("data/swot"):
        for file in files:
            if data_str in file and file.endswith(".nc"):
                swot_files.append(os.path.join(root, file))

    # Encontrar arquivos MODIS para a data
    modis_files = []
    for root, dirs, files in os.walk("data/modis"):
        for file in files:
            if data_str in file and file.endswith(".nc"):
                modis_files.append(os.path.join(root, file))

    print(
        f"  Data {data}: {len(swot_files)} arquivos SWOT, {len(modis_files)} arquivos MODIS"
    )

    # Carregar dados SWOT
    swot_lats, swot_lons, swot_ssha = carregar_dados_swot(swot_files)

    # Carregar dados MODIS
    modis_lats, modis_lons, modis_chlor = carregar_dados_modis(modis_files)

    return swot_lats, swot_lons, swot_ssha, modis_lats, modis_lons, modis_chlor


def descobrir_datas_disponiveis():
    """
    Descobre todas as datas disponíveis nos arquivos MODIS e SWOT.

    Returns:
        list: lista de datas no formato YYYY-MM-DD
    """
    datas = set()

    # Procurar arquivos MODIS
    for root, dirs, files in os.walk("data/modis"):
        for file in files:
            if file.endswith(".nc") and "2024" in file:
                # Extrair data do nome do arquivo
                try:
                    data_str = file.split(".")[1]  # AQUA_MODIS.20240101.L3b.DAY.CHL.nc
                    data_obj = datetime.strptime(data_str, "%Y%m%d")
                    datas.add(data_obj.strftime("%Y-%m-%d"))
                except:
                    continue

    # Procurar arquivos SWOT
    for root, dirs, files in os.walk("data/swot"):
        for file in files:
            if file.endswith(".nc") and "2024" in file:
                # Extrair data do nome do arquivo
                try:
                    # SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc
                    parts = file.split("_")
                    for part in parts:
                        if "2024" in part and len(part) >= 8:
                            data_str = part[:8]  # 20240101
                            data_obj = datetime.strptime(data_str, "%Y%m%d")
                            datas.add(data_obj.strftime("%Y-%m-%d"))
                            break
                except:
                    continue

    return sorted(list(datas))


def simular_tubarao_por_data(id_tubarao, data, df_ambiental, kdtree, coords):
    """
    Simula um tubarão para uma data específica.

    Args:
        id_tubarao: ID único do tubarão
        data: data no formato YYYY-MM-DD
        df_ambiental: DataFrame com dados ambientais
        kdtree: KDTree para busca espacial
        coords: array de coordenadas

    Returns:
        list: lista de registros do tubarão
    """
    registros = []

    # Converter data para datetime
    data_obj = datetime.strptime(data, "%Y-%m-%d")

    # Selecionar ponto inicial aleatório
    idx_inicial = np.random.randint(0, len(df_ambiental))
    ponto_inicial = df_ambiental.iloc[idx_inicial]

    lat_atual = ponto_inicial["lat"]
    lon_atual = ponto_inicial["lon"]
    ssha_atual = ponto_inicial["ssha"]
    chlor_a_atual = ponto_inicial["chlor_a"]

    # Simular movimento ao longo do dia
    for ping in range(PINGS_POR_TUBARAO):
        # Calcular tempo atual
        tempo_atual = data_obj + timedelta(minutes=ping * INTERVALO_PING_MINUTOS)
        tempo_dia = tempo_atual.hour + tempo_atual.minute / 60.0

        # Calcular nível de fome (varia ao longo do dia)
        nivel_fome = 0.3 + 0.4 * np.sin(2 * np.pi * tempo_dia / 24)

        # Calcular profundidade baseada no comportamento
        profundidade_m = calcular_profundidade_comportamental(
            tempo_dia, ssha_atual, chlor_a_atual
        )

        # Simular dados dos sensores
        dados_sensores = simular_dados_sensores(profundidade_m, tempo_dia, nivel_fome)

        # Converter coordenadas para telemetria
        lat_int, lon_int = converter_coordenadas_para_telemetria(lat_atual, lon_atual)

        # Calcular CRC-16 (simplificado - usar apenas valores básicos)
        # Criar string simples para calcular CRC
        crc_string = f"{id_tubarao}{int(tempo_atual.timestamp())}{lat_int}{lon_int}"
        data_bytes = crc_string.encode("utf-8")
        crc16 = calcular_crc16_ccitt(data_bytes)

        # Registrar dados de telemetria (SEM p_forrageio e comportamento)
        registro = {
            "id": id_tubarao,
            "timestamp": int(tempo_atual.timestamp()),
            "lat": lat_int,
            "lon": lon_int,
            "depth_dm": dados_sensores["depth_dm"],
            "temp_cC": dados_sensores["temp_cC"],
            "batt_mV": dados_sensores["batt_mV"],
            "acc_x": dados_sensores["acc_x"],
            "acc_y": dados_sensores["acc_y"],
            "acc_z": dados_sensores["acc_z"],
            "gyr_x": dados_sensores["gyr_x"],
            "gyr_y": dados_sensores["gyr_y"],
            "gyr_z": dados_sensores["gyr_z"],
            "crc16": crc16,
        }
        registros.append(registro)

        # Simular movimento (pequena variação)
        lat_atual += np.random.normal(0, 0.001)  # ~100m
        lon_atual += np.random.normal(0, 0.001)

        # Manter dentro de limites oceânicos
        lat_atual = max(-90, min(90, lat_atual))
        lon_atual = max(-180, min(180, lon_atual))

    return registros


def unir_dados_ambientais(
    df_tubaroes, swot_lats, swot_lons, swot_ssha, modis_lats, modis_lons, modis_chlor
):
    """
    Une dados de tubarões com dados ambientais (SWOT + MODIS).

    Args:
        df_tubaroes: DataFrame com dados dos tubarões
        swot_lats, swot_lons, swot_ssha: dados SWOT
        modis_lats, modis_lons, modis_chlor: dados MODIS

    Returns:
        DataFrame com dados unificados
    """
    print(f"    Unindo dados ambientais...")

    # Construir árvores KD para busca espacial eficiente
    swot_kdtree = cKDTree(np.column_stack([swot_lats, swot_lons]))
    modis_kdtree = cKDTree(np.column_stack([modis_lats, modis_lons]))

    dados_unificados = []

    for _, row in tqdm(
        df_tubaroes.iterrows(), total=len(df_tubaroes), desc="      Unindo", leave=False
    ):
        # Converter coordenadas de telemetria para graus decimais
        lat_tubarao = row["lat"] / 10000.0  # Converter de graus × 1e-4 para graus
        lon_tubarao = row["lon"] / 10000.0  # Converter de graus × 1e-4 para graus
        query_point = np.array([[lat_tubarao, lon_tubarao]])

        # Buscar SWOT mais próximo
        ssha_ambiente = np.nan
        if len(swot_lats) > 0:
            dist_swot, idx_swot = swot_kdtree.query(query_point, k=1)
            if dist_swot[0] < 1.0:  # Tolerância de 1 grau
                ssha_ambiente = swot_ssha[idx_swot[0]]

        # Buscar MODIS mais próximo
        chlor_a_ambiente = np.nan
        if len(modis_lats) > 0:
            dist_modis, idx_modis = modis_kdtree.query(query_point, k=1)
            if dist_modis[0] < 1.0:  # Tolerância de 1 grau
                chlor_a_ambiente = modis_chlor[idx_modis[0]]

        # Criar registro unificado
        registro = {
            "id": row["id"],
            "timestamp": row["timestamp"],
            "lat": row["lat"],
            "lon": row["lon"],
            "depth_dm": row["depth_dm"],
            "temp_cC": row["temp_cC"],
            "batt_mV": row["batt_mV"],
            "acc_x": row["acc_x"],
            "acc_y": row["acc_y"],
            "acc_z": row["acc_z"],
            "gyr_x": row["gyr_x"],
            "gyr_y": row["gyr_y"],
            "gyr_z": row["gyr_z"],
            "crc16": row["crc16"],
            "ssha_ambiente": ssha_ambiente,
            "chlor_a_ambiente": chlor_a_ambiente,
        }
        dados_unificados.append(registro)

    return pd.DataFrame(dados_unificados)


def main():
    """Função principal para gerar dados de validação."""
    print("=== GERADOR DE DADOS DE VALIDACAO PARA IA ===")
    print("Gerando dados sinteticos de tubaroes para validacao da IA")
    print("SEM colunas p_forrageio e comportamento")
    print("COM uniao de dados ambientais (SWOT + MODIS)")
    print()

    # Descobrir datas disponíveis
    datas_disponiveis = descobrir_datas_disponiveis()
    print(f"Datas disponiveis: {len(datas_disponiveis)}")
    for data in datas_disponiveis[:5]:  # Mostrar primeiras 5
        print(f"  - {data}")
    if len(datas_disponiveis) > 5:
        print(f"  ... e mais {len(datas_disponiveis) - 5} datas")
    print()

    print(f"Simulando {N_TUBAROES} tubaroes por data ({PINGS_POR_TUBARAO} pings cada)")

    # Gerar dados ambientais sintéticos para pontos iniciais
    df_ambiental, kdtree, coords = gerar_dados_ambientais_sinteticos()
    print(f"Usando {len(df_ambiental):,} pontos como base ambiental")

    # Selecionar pontos iniciais aleatórios
    np.random.seed(42)  # Para reprodutibilidade
    indices_iniciais = np.random.choice(len(df_ambiental), N_TUBAROES, replace=False)
    pontos_iniciais = [
        {"lat": df_ambiental.iloc[idx]["lat"], "lon": df_ambiental.iloc[idx]["lon"]}
        for idx in indices_iniciais
    ]

    # Preparar diretório de saída
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Simular para cada data
    todos_registros = []

    print("\nIniciando simulacao por data...")
    for data in tqdm(datas_disponiveis, desc="Simulando por data"):
        print(f"\nProcessando {data}...")

        # Carregar dados ambientais reais para esta data
        swot_lats, swot_lons, swot_ssha, modis_lats, modis_lons, modis_chlor = (
            carregar_dados_ambientais_por_data(data)
        )

        # Se não houver dados ambientais reais, usar sintéticos
        if len(swot_lats) == 0 or len(modis_lats) == 0:
            print(f"  Usando dados ambientais sinteticos para {data}")
            swot_lats = df_ambiental["lat"].values
            swot_lons = df_ambiental["lon"].values
            swot_ssha = df_ambiental["ssha"].values
            modis_lats = df_ambiental["lat"].values
            modis_lons = df_ambiental["lon"].values
            modis_chlor = df_ambiental["chlor_a"].values

        registros_dia = []

        for i, ponto_inicial in enumerate(pontos_iniciais):
            id_tubarao = i + 1
            registros_tubarao = simular_tubarao_por_data(
                id_tubarao, data, df_ambiental, kdtree, coords
            )
            registros_dia.extend(registros_tubarao)

        # Unir dados dos tubarões com dados ambientais
        df_tubaroes_dia = pd.DataFrame(registros_dia)
        df_unificado_dia = unir_dados_ambientais(
            df_tubaroes_dia,
            swot_lats,
            swot_lons,
            swot_ssha,
            modis_lats,
            modis_lons,
            modis_chlor,
        )

        todos_registros.extend(df_unificado_dia.to_dict("records"))

    # Criar DataFrame final
    df_final = pd.DataFrame(todos_registros)

    # Salvar arquivo de validação
    arquivo_validacao = os.path.join(OUTPUT_DIR, "dados_validacao.csv")
    df_final.to_csv(arquivo_validacao, index=False)

    # Estatísticas finais
    print(f"\n=== ESTATISTICAS FINAIS ===")
    print(f"Total de registros gerados: {len(df_final):,}")
    print(f"Tubaroes simulados: {N_TUBAROES}")
    print(f"Datas processadas: {len(datas_disponiveis)}")
    print(f"Pings por tubarao por dia: {PINGS_POR_TUBARAO}")
    print(f"Arquivo salvo: {arquivo_validacao}")
    print()

    print("Colunas geradas:")
    for col in df_final.columns:
        print(f"  - {col}")
    print()

    print("Estatisticas dos dados:")
    print(f"  - Profundidade media: {df_final['depth_dm'].mean()/10:.1f}m")
    print(f"  - Profundidade maxima: {df_final['depth_dm'].max()/10:.1f}m")
    print(f"  - Temperatura media: {df_final['temp_cC'].mean()/100:.1f}C")
    print(f"  - Bateria media: {df_final['batt_mV'].mean():.0f}mV")
    print(f"  - CRC-16 validos: {len(df_final[df_final['crc16'] > 0]):,}")
    print(
        f"  - SSHA ambiente validos: {len(df_final[~df_final['ssha_ambiente'].isna()]):,}"
    )
    print(
        f"  - Chlor_a ambiente validos: {len(df_final[~df_final['chlor_a_ambiente'].isna()]):,}"
    )

    print(f"\nDados de validacao gerados com sucesso!")
    print(f"Arquivo: {arquivo_validacao}")


if __name__ == "__main__":
    main()
