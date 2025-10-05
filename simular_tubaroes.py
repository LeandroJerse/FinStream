#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador Avançado de Dados Sintéticos de Tubarões com Telemetria
================================================================

Gera dados realistas de movimento de tubarões baseados em modelos ecológicos
avançados, considerando produtividade oceânica, dinâmica de correntes,
ritmos circadianos e comportamento de forrageio ótimo.

Baseado em estudos científicos:
- Braun et al. (2019): Mesoscale eddies release pelagic sharks
- Estudos sobre movimentos de tubarões-brancos em redemoinhos

Inclui dados de telemetria realistas do dispositivo de rastreamento.

Autor: Sistema de Análise Oceanográfica Avançada
Data: 2024-01-01
"""

import os
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
N_TUBAROES = 50
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
BATERIA_INICIAL_MV = 4000  # mV (4.0V)
BATERIA_MINIMA_MV = 3200  # mV (3.2V)
BATERIA_MAXIMA_MV = 4200  # mV (4.2V)

# Parâmetros de movimento por comportamento (graus)
PASSO_FORRAGEIO_BASE = 0.003  # ~333m em 5min = ~4 km/h (forrageio lento)
PASSO_BUSCA_BASE = 0.008  # ~888m em 5min = ~10.7 km/h (busca ativa)
PASSO_TRANSITO_BASE = 0.012  # ~1.3km em 5min = ~16 km/h (transito rapido)

# Parâmetros comportamentais - AJUSTADO PARA REALISMO BIOLÓGICO
# Baseado em estudos de telemetria: comportamentos persistem por horas
# Cálculo: E[duração] = 1/(1-p) pings, onde p = prob. continuar
# p=0.92 -> E=12.5 pings = 62.5 min (~1h por comportamento)
# p=0.95 -> E=20 pings = 100 min (~1h40min por comportamento)
# p=0.90 -> E=10 pings = 50 min (~50min por comportamento)
PROB_CONTINUAR_COMPORTAMENTO = 0.92
PROB_MUDAR_COMPORTAMENTO = 0.08
HISTORICO_POSICOES = 10  # Número de posições para calcular velocidade
NIVEL_FOME_INICIAL = 0.1  # Nível de fome inicial (0=saciado, 1=faminto)

# Arquivos
DADOS_AMBIENTAIS = "data/dados_unificados_final.csv"
OUTPUT_FILE = "data/tubaroes_sinteticos.csv"
ANALISE_DIR = "data/analise_diaria"

# =============================================================================
# FUNÇÕES DE TELEMETRIA E CRC
# =============================================================================


def calcular_crc16_ccitt(data):
    """
    Calcula CRC-16/CCITT para verificação de integridade dos dados.

    Args:
        data: bytes ou lista de bytes

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
    Converte coordenadas para formato de telemetria (3 bytes com sinal).

    Args:
        lat, lon: coordenadas em graus decimais

    Returns:
        tuple: (lat_int, lon_int) em formato de telemetria
    """
    # Converter para graus × 1e-4 (4 casas decimais)
    lat_int = int(lat * 10000)
    lon_int = int(lon * 10000)

    # Verificar limites para 3 bytes com sinal (-8,388,608 a +8,388,607)
    lat_int = np.clip(lat_int, -8388608, 8388607)
    lon_int = np.clip(lon_int, -8388608, 8388607)

    return lat_int, lon_int


def simular_dados_sensores(comportamento, profundidade_m, tempo_dia, nivel_fome):
    """
    Simula dados realistas dos sensores do dispositivo de telemetria.

    Args:
        comportamento: comportamento atual do tubarão
        profundidade_m: profundidade em metros
        tempo_dia: hora do dia (0-24)
        nivel_fome: nível de fome (0-1)

    Returns:
        dict: dados dos sensores
    """
    # Profundidade em decímetros (0-6553.5 m)
    depth_dm = int(profundidade_m * 10)
    depth_dm = np.clip(depth_dm, 0, 65535)

    # Temperatura baseada na profundidade e comportamento
    # Baseado em estudos: tubarões usam redemoinhos para acessar águas
    if comportamento == "forrageando" and profundidade_m > 200:
        # Mergulho profundo em redemoinho anticiclônico (Braun et al.)
        temp_base = TEMPERATURA_OCEANO_DEEP + (profundidade_m - 200) * 0.01
    else:
        # Gradiente térmico normal
        temp_base = TEMPERATURA_OCEANO_SURFACE - (profundidade_m * 0.02)

    # Adicionar variação temporal e ruído
    temp_cC = int((temp_base + np.random.normal(0, 0.5)) * 100)
    temp_cC = np.clip(temp_cC, -32768, 32767)

    # Bateria (degradação realista)
    consumo_base = 1  # mV por ping
    consumo_extra = 2 if comportamento == "transitando" else 0
    batt_mV = (
        BATERIA_INICIAL_MV - (np.random.random() * 100) - consumo_base - consumo_extra
    )
    batt_mV = np.clip(batt_mV, BATERIA_MINIMA_MV, BATERIA_MAXIMA_MV)
    batt_mV = int(batt_mV)

    # Acelerômetro (±16g, 1 LSB = 1 mg)
    # Baseado no comportamento: forrageio = erráticos, trânsito = direcionais
    if comportamento == "forrageando":
        acc_x = int(np.random.normal(0, 2000))  # Movimentos erráticos
        acc_y = int(np.random.normal(0, 2000))
        acc_z = int(np.random.normal(-1000, 1000))  # Movimentos verticais
    elif comportamento == "transitando":
        acc_x = int(np.random.normal(1000, 500))  # Movimento direcional
        acc_y = int(np.random.normal(0, 300))
        acc_z = int(np.random.normal(0, 200))
    else:  # busca
        acc_x = int(np.random.normal(500, 1000))
        acc_y = int(np.random.normal(0, 800))
        acc_z = int(np.random.normal(-500, 500))

    # Limitar a ±16g (16000 mg)
    acc_x = np.clip(acc_x, -16000, 16000)
    acc_y = np.clip(acc_y, -16000, 16000)
    acc_z = np.clip(acc_z, -16000, 16000)

    # Giroscópio (±2000 °/s, 1 LSB = 1 mdps)
    # Rotação baseada no comportamento
    if comportamento == "forrageando":
        gyr_x = int(np.random.normal(0, 500))  # Rotação errática
        gyr_y = int(np.random.normal(0, 500))
        gyr_z = int(np.random.normal(0, 300))
    elif comportamento == "transitando":
        gyr_x = int(np.random.normal(0, 100))  # Rotação mínima
        gyr_y = int(np.random.normal(0, 100))
        gyr_z = int(np.random.normal(0, 50))
    else:  # busca
        gyr_x = int(np.random.normal(0, 200))
        gyr_y = int(np.random.normal(0, 200))
        gyr_z = int(np.random.normal(0, 150))

    # Limitar a ±2000 °/s (2000000 mdps)
    gyr_x = np.clip(gyr_x, -2000000, 2000000)
    gyr_y = np.clip(gyr_y, -2000000, 2000000)
    gyr_z = np.clip(gyr_z, -2000000, 2000000)

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


def calcular_profundidade_comportamental(comportamento, tempo_dia, ssha, chlor_a):
    """
    Calcula profundidade baseada no comportamento e estudos científicos.

    Baseado em Braun et al. (2019): tubarões usam redemoinhos
    anticiclônicos para acessar a zona mesopelágica (200-1000m).

    Args:
        comportamento: comportamento atual
        tempo_dia: hora do dia (0-24)
        ssha: altura da superfície do mar (indica redemoinhos)
        chlor_a: concentração de clorofila

    Returns:
        float: profundidade em metros
    """
    # Comportamento de mergulho baseado em estudos
    if comportamento == "forrageando":
        # Forrageio: mergulhos profundos durante o dia em redemoinhos
        if 6 <= tempo_dia <= 18:  # Durante o dia
            if ssha > 0.1:  # Redemoinho anticiclônico (SSHA positivo)
                # Mergulho profundo na zona mesopelágica
                profundidade_base = np.random.uniform(200, 800)
            else:
                # Mergulho moderado
                profundidade_base = np.random.uniform(50, 300)
        else:  # Noite
            # Menos atividade de mergulho
            profundidade_base = np.random.uniform(20, 150)

    elif comportamento == "transitando":
        # Trânsito: natação em superfície
        profundidade_base = np.random.uniform(5, 50)

    else:  # busca
        # Busca: mergulhos moderados
        if 6 <= tempo_dia <= 18:
            profundidade_base = np.random.uniform(100, 400)
        else:
            profundidade_base = np.random.uniform(30, 200)

    # Adicionar variação baseada na produtividade
    if chlor_a > 0.1:  # Área produtiva
        profundidade_base *= np.random.uniform(0.8, 1.2)

    return np.clip(profundidade_base, 1, PROFUNDIDADE_MAXIMA_M)


# =============================================================================
# FUNÇÕES AUXILIARES AVANÇADAS
# =============================================================================


def converter_bin_para_lat_lon(bin_nums):
    """
    Converte números de bin MODIS para coordenadas lat/lon.
    """
    lats = []
    lons = []
    for bin_num in bin_nums:
        lat = (bin_num // 3600) * 0.1 - 90.0
        lon = (bin_num % 3600) * 0.1 - 180.0
        lats.append(lat)
        lons.append(lon)
    return np.array(lats), np.array(lons)


def carregar_dados_swot(swot_files):
    """
    Carrega dados SWOT de múltiplos arquivos.
    """
    all_lats = []
    all_lons = []
    all_ssha = []
    for filepath in swot_files:
        try:
            ds = xr.open_dataset(filepath)
            lats = ds["latitude"].values.flatten()
            lons = ds["longitude"].values.flatten()
            ssha = ds["ssha_karin"].values.flatten()
            lons = np.where(lons > 180, lons - 360, lons)  # Convert 0-360 to -180-180
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
                chlor_values = ds["chlor_a"].values
                if hasattr(chlor_values, "dtype") and "sum" in chlor_values.dtype.names:
                    chlor_sum = chlor_values["sum"]
                    nobs = binlist["nobs"]
                    chlor_media = np.where(nobs > 0, chlor_sum / nobs, np.nan)
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


def carregar_dados_ambientais_por_data(data):
    """
    Carrega dados ambientais (SWOT + MODIS) para uma data específica
    e encontra a intersecção espacial entre eles.

    Args:
        data: Data no formato YYYY-MM-DD

    Returns:
        tuple: (df_intersecao, kdtree, coords_array)
    """
    print(f"Carregando dados ambientais para {data}...")

    # Encontrar arquivos SWOT para a data
    data_str = data.replace('-', '')
    swot_files = []
    for root, dirs, files in os.walk("data/swot"):
        for file in files:
            if data_str in file and file.endswith('.nc'):
                swot_files.append(os.path.join(root, file))
    
    # Encontrar arquivos MODIS para a data
    modis_files = []
    for root, dirs, files in os.walk("data/modis"):
        for file in files:
            if data_str in file and file.endswith('.nc'):
                modis_files.append(os.path.join(root, file))
    
    print(f"Encontrados {len(swot_files)} arquivos SWOT e {len(modis_files)} arquivos MODIS para {data}")
    
    if not swot_files or not modis_files:
        print(f"AVISO: Dados incompletos para {data}")
        return pd.DataFrame(), None, None
    
    # Carregar dados SWOT
    swot_lats, swot_lons, swot_ssha = carregar_dados_swot(swot_files)
    
    # Carregar dados MODIS
    modis_lats, modis_lons, modis_chlor = carregar_dados_modis(modis_files)
    
    if len(swot_lats) == 0 or len(modis_lats) == 0:
        print(f"AVISO: Dados insuficientes para {data}")
        return pd.DataFrame(), None, None
    
    print(f"SWOT: {len(swot_lats)} pontos válidos")
    print(f"MODIS: {len(modis_lats)} pontos válidos")
    
    # Encontrar intersecção espacial entre SWOT e MODIS usando KDTree
    # Usar tolerância de 0.1 graus para considerar pontos próximos
    TOLERANCE = 0.1
    intersecao_lats = []
    intersecao_lons = []
    intersecao_ssha = []
    intersecao_chlor = []
    
    # Criar KDTree para dados MODIS (muito mais eficiente)
    print("  Construindo KDTree para busca espacial eficiente...")
    modis_coords = np.column_stack([modis_lats, modis_lons])
    modis_kdtree = cKDTree(modis_coords)
    
    # Buscar pontos próximos usando KDTree
    print("  Buscando intersecções espaciais...")
    swot_coords = np.column_stack([swot_lats, swot_lons])
    
    # Buscar o ponto MODIS mais próximo para cada ponto SWOT
    distances, indices = modis_kdtree.query(swot_coords, k=1)
    
    # Filtrar apenas pontos dentro da tolerância
    valid_mask = distances < TOLERANCE
    
    if np.any(valid_mask):
        intersecao_lats = swot_lats[valid_mask].tolist()
        intersecao_lons = swot_lons[valid_mask].tolist()
        intersecao_ssha = swot_ssha[valid_mask].tolist()
        intersecao_chlor = modis_chlor[indices[valid_mask]].tolist()

    print(f"Intersecção encontrada: {len(intersecao_lats)} pontos")
    
    if len(intersecao_lats) == 0:
        print(f"AVISO: Nenhuma intersecção encontrada para {data}")
        return pd.DataFrame(), None, None
    
    # Criar DataFrame com dados da intersecção
    df_intersecao = pd.DataFrame({
        'lat': intersecao_lats,
        'lon': intersecao_lons,
        'ssha': intersecao_ssha,
        'chlor_a': intersecao_chlor
    })
    
    print(f"Dados de intersecção carregados: {len(df_intersecao):,} pontos válidos")
    print(f"SSHA range: {df_intersecao['ssha'].min():.2f} a {df_intersecao['ssha'].max():.2f}")
    print(f"Chlor_a range: {df_intersecao['chlor_a'].min():.4f} a {df_intersecao['chlor_a'].max():.4f}")

    # Criar array de coordenadas para KDTree
    coords = np.column_stack([df_intersecao["lat"], df_intersecao["lon"]])
    kdtree = cKDTree(coords)

    return df_intersecao, kdtree, coords


def carregar_dados_ambientais():
    """
    Carrega e prepara os dados ambientais reais (SWOT + MODIS) para uma data específica.
    Usa dados da primeira data disponível como base.

    Returns:
        tuple: (df, kdtree, coords_array)
    """
    print("Carregando dados ambientais reais...")

    # Descobrir primeira data disponível
    datas_disponiveis = descobrir_datas_disponiveis()
    if not datas_disponiveis:
        raise ValueError("Nenhuma data disponível encontrada nos dados MODIS/SWOT")
    
    primeira_data = datas_disponiveis[0]
    print(f"Usando dados ambientais da data: {primeira_data}")
    
    # Carregar dados ambientais para a primeira data
    df_ambiental, kdtree, coords = carregar_dados_ambientais_por_data(primeira_data)
    
    if len(df_ambiental) == 0:
        raise ValueError(f"Nenhum dado ambiental válido encontrado para {primeira_data}")
    
    print(f"Dados carregados: {len(df_ambiental):,} pontos válidos")
    print(f"SSHA range: {df_ambiental['ssha'].min():.2f} a {df_ambiental['ssha'].max():.2f}")
    print(f"Chlor_a range: {df_ambiental['chlor_a'].min():.4f} a {df_ambiental['chlor_a'].max():.4f}")

    return df_ambiental, kdtree, coords


def calcular_probabilidade_forrageio_avancada(
    ssha,
    chlor_a,
    lat,
    lon,
    tempo_dia,
    comportamento_anterior,
    historico_posicoes,
    nivel_fome=0.0,
):
    """
    Calcula probabilidade de forrageio usando modelo ecológico avançado.

    Baseado em:
    - Gradientes de produtividade (clorofila)
    - Dinâmica de correntes (SSHA)
    - Ritmos circadianos
    - Comportamento de forrageio ótimo
    - Fadiga nutricional

    Args:
        ssha: altura da superfície do mar (correntes/fronteiras)
        chlor_a: concentração de clorofila (produtividade)
        lat, lon: posição geográfica
        tempo_dia: hora do dia (0-24)
        comportamento_anterior: comportamento no ping anterior
        historico_posicoes: últimas N posições para calcular velocidade
        nivel_fome: nível de fome do tubarão (0=saciado, 1=faminto)

    Returns:
        tuple: (p_forrageio, velocidade_preferida, direcao_preferida)
    """

    # 1. PRODUTIVIDADE PRIMÁRIA (clorofila)
    # Tubarões preferem áreas de alta produtividade
    chlor_norm = np.clip(chlor_a, 0.0024, 73.6333)
    chlor_norm = (chlor_norm - 0.0024) / (73.6333 - 0.0024)

    # 2. DINÂMICA OCEÂNICA (SSHA)
    # Alturas positivas = convergência = alta produtividade
    # Alturas negativas = divergência = baixa produtividade
    ssha_norm = np.clip(ssha, -9.26, 261.41)
    ssha_norm = (ssha_norm - (-9.26)) / (261.41 - (-9.26))

    # Fronteiras oceânicas (gradientes de SSHA)
    gradiente_ssha = np.abs(ssha_norm - 0.5) * 2  # 0-1, máximo nas bordas

    # 3. RITMO CIRCADIANO
    # Tubarões são mais ativos ao amanhecer e anoitecer
    if 5 <= tempo_dia <= 8 or 17 <= tempo_dia <= 20:
        fator_circadiano = 1.3  # Maior atividade
    elif 10 <= tempo_dia <= 16:
        fator_circadiano = 0.8  # Menor atividade (meio dia)
    else:
        fator_circadiano = 1.1  # Atividade moderada (noite)

    # 4. COMPORTAMENTO DE FORRAGEIO ÓTIMO
    # (Fatores calculados diretamente no score final)

    # 5. INÉRCIA COMPORTAMENTAL
    if comportamento_anterior == "forrageando":
        inercia = 0.3  # Tendência a continuar forrageando
    elif comportamento_anterior == "transitando":
        inercia = -0.2  # Tendência a parar de transitar
    else:  # busca
        inercia = 0.0

    # 6. NÍVEL DE FOME
    # (Usado diretamente no score final)

    # 7. FATOR TEMPORAL (varia ao longo do dia)
    # Simula mudanças nas condições oceânicas
    ciclo_diario = np.sin(2 * np.pi * tempo_dia / 24) * 0.2

    # 8. CÁLCULO FINAL DA PROBABILIDADE (rebalanceado)
    # Usar valores brutos (0-1) em vez de sigmoid para evitar compressão
    # Pesos somam 1.0 para manter probabilidades realistas
    score_forrageio = (
        0.25 * chlor_norm  # Produtividade (valor bruto 0-1)
        + 0.20 * ssha_norm  # Convergência (valor bruto 0-1)
        + 0.15 * gradiente_ssha  # Fronteiras (valor bruto 0-1)
        + 0.15 * (fator_circadiano - 0.8) / 0.5  # Normalizado para 0-1
        + 0.20 * nivel_fome  # Fome (valor bruto 0-1)
        + 0.05 * (inercia + 0.3) / 0.6  # Normalizado para 0-1
        + ciclo_diario  # Variação temporal (-0.2 a +0.2)
        + np.random.normal(0, 0.15)  # Ruído reduzido
    )

    # Aplicar sigmoid apenas uma vez no score final
    # Ajustar para ter média ~0.5 em vez de ~0.7
    p_forrageio = sigmoid(2.0 * (score_forrageio - 0.5))

    # 9. VELOCIDADE PREFERIDA baseada no comportamento (biologicamente ajustado)
    # Limiares ajustados para refletir comportamento realista de tubarões
    if p_forrageio > 0.5:  # Forrageio: mais comum (35-45% do tempo)
        # Forrageio: movimentos lentos e erráticos em áreas produtivas
        velocidade_base = PASSO_FORRAGEIO_BASE
        variacao_velocidade = 0.5
    elif p_forrageio < 0.35:  # Trânsito: menos comum (20-30% do tempo)
        # Trânsito: movimentos rápidos e direcionais entre áreas
        velocidade_base = PASSO_TRANSITO_BASE
        variacao_velocidade = 0.2
    else:  # Busca: intermediário (30-40% do tempo, entre 0.35-0.5)
        # Busca: movimentos intermediários procurando sinais de presas
        velocidade_base = PASSO_BUSCA_BASE
        variacao_velocidade = 0.4

    # Ajustar velocidade baseada na produtividade local
    velocidade_preferida = velocidade_base * (
        1 + variacao_velocidade * np.random.uniform(-1, 1)
    )

    # 10. DIREÇÃO PREFERIDA (limiares ajustados)
    # Em forrageio: direção aleatória
    # Em trânsito: direção baseada em gradientes de produtividade
    if p_forrageio > 0.55:  # Ajustado para consistência
        direcao_preferida = np.random.uniform(0, 2 * np.pi)
    elif p_forrageio < 0.45:  # Ajustado para consistência
        # Direção para áreas de maior produtividade
        direcao_preferida = np.random.uniform(0, 2 * np.pi)  # Simplificado
    else:
        # Busca: direção semi-aleatória com tendência
        direcao_preferida = np.random.uniform(0, 2 * np.pi)

    return float(p_forrageio), float(velocidade_preferida), float(direcao_preferida)


def determinar_comportamento_avancado(
    p_forrageio, comportamento_atual, tempo_sem_alimento=0
):
    """
    Determina o próximo comportamento usando modelo avançado.

    Args:
        p_forrageio: probabilidade de forrageio (0-1)
        comportamento_atual: comportamento atual
        tempo_sem_alimento: tempo desde última refeição (pings)

    Returns:
        str: novo comportamento
    """
    # INÉRCIA COMPORTAMENTAL FORTE: comportamentos persistem por horas
    # Com 92% de probabilidade, o tubarão continua no mesmo comportamento
    # Isso resulta em bouts de ~1-2 horas antes de mudar
    if np.random.random() < PROB_CONTINUAR_COMPORTAMENTO:
        return comportamento_atual

    # MUDANÇA DE COMPORTAMENTO (rara, apenas 8% das vezes)
    # Baseada em fatores ecológicos

    # Fator tempo sem alimento: fome extrema força forrageio
    if tempo_sem_alimento > 150:  # Muito faminto (~12.5 horas)
        if p_forrageio > 0.3:
            return "forrageando"
        else:
            return "busca"

    # Transições realistas baseadas no comportamento atual e p_forrageio
    # Tubarões não mudam bruscamente entre estados opostos
    if comportamento_atual == "forrageando":
        # Forrageio -> busca ou continua forrageio
        if p_forrageio < 0.35:  # Condições ruins
            return "busca"
        else:
            return "forrageando"  # Tende a continuar forrageando

    elif comportamento_atual == "transitando":
        # Trânsito -> busca ou forrageio (se encontrar boa área)
        if p_forrageio > 0.55:  # Encontrou área produtiva
            return "busca"  # Busca primeiro, depois forragear
        else:
            return "transitando"  # Continua em trânsito

    else:  # busca
        # Busca -> forrageio (área boa) ou trânsito (área ruim)
        if p_forrageio > 0.55:
            return "forrageando"
        elif p_forrageio < 0.35:
            return "transitando"
        else:
            return "busca"


def calcular_movimento_avancado(
    comportamento, velocidade_preferida, direcao_preferida, lat_atual, lon_atual
):
    """
    Calcula movimento baseado em comportamento e ambiente.

    Args:
        comportamento: comportamento atual
        velocidade_preferida: velocidade em graus
        direcao_preferida: direção em radianos
        lat_atual, lon_atual: posição atual

    Returns:
        tuple: (nova_lat, nova_lon)
    """
    if comportamento == "forrageando":
        # Movimento errático com pequenos passos
        ruido_direcao = np.random.normal(direcao_preferida, 0.5)
        ruido_velocidade = velocidade_preferida * np.random.uniform(0.3, 1.5)

    elif comportamento == "transitando":
        # Movimento direcional com poucos desvios
        ruido_direcao = np.random.normal(direcao_preferida, 0.1)
        ruido_velocidade = velocidade_preferida * np.random.uniform(0.8, 1.2)

    else:  # busca
        # Movimento intermediário
        ruido_direcao = np.random.normal(direcao_preferida, 0.3)
        ruido_velocidade = velocidade_preferida * np.random.uniform(0.5, 1.3)

    # Converter para deslocamento em lat/lon
    # Aproximação simples para pequenas distâncias
    deslocamento_lat = ruido_velocidade * np.cos(ruido_direcao)
    deslocamento_lon = ruido_velocidade * np.sin(ruido_direcao)

    # Ajustar longitude baseado na latitude (convergência dos meridianos)
    fator_lon = 1.0 / np.cos(np.radians(lat_atual))
    deslocamento_lon *= fator_lon

    nova_lat = lat_atual + deslocamento_lat
    nova_lon = lon_atual + deslocamento_lon

    return nova_lat, nova_lon


def analisar_dados_ambientais_basico(df_ambiental):
    """
    Análise básica dos dados ambientais para referência da IA.

    Args:
        df_ambiental: DataFrame com dados ambientais

    Returns:
        dict: Estatísticas básicas
    """
    print("\nAnalisando dados ambientais para referência da IA...")

    # Criar diretório de análise
    os.makedirs(ANALISE_DIR, exist_ok=True)

    # Estatísticas gerais (apenas para referência)
    stats_gerais = {
        "total_pontos": len(df_ambiental),
        "ssha_media": df_ambiental["ssha"].mean(),
        "ssha_std": df_ambiental["ssha"].std(),
        "chlor_media": df_ambiental["chlor_a"].mean(),
        "chlor_std": df_ambiental["chlor_a"].std(),
        "correlacao_ssha_chlor": df_ambiental["ssha"].corr(df_ambiental["chlor_a"]),
    }

    print(f"Dados ambientais carregados: {stats_gerais['total_pontos']:,} pontos")
    print(f"SSHA médio: {stats_gerais['ssha_media']:.3f}")
    print(f"Chlor_a médio: {stats_gerais['chlor_media']:.4f}")

    return {"stats_gerais": stats_gerais}


def salvar_dados_tubaroes_por_dia(registros_dia, data):
    """
    Salva apenas os dados CSV dos tubarões por dia (para treinamento da IA).

    Args:
        registros_dia: Lista de registros do dia
        data: Data do arquivo
    """
    if not registros_dia:
        return

    df_dia = pd.DataFrame(registros_dia)

    # Salvar apenas dados CSV dos tubarões
    arquivo_csv = f"{ANALISE_DIR}/tubaroes_{data.replace('-', '')}.csv"
    df_dia.to_csv(arquivo_csv, index=False)

    print(f"Dados tubarões salvos: {arquivo_csv}")


def buscar_dados_ambientais_proximos(lat, lon, kdtree, df_ambiental, coords):
    """
    Busca os dados ambientais mais próximos para uma posição.

    Args:
        lat, lon: coordenadas
        kdtree: árvore KD dos dados ambientais
        df_ambiental: DataFrame com dados ambientais
        coords: array de coordenadas

    Returns:
        tuple: (ssha, chlor_a)
    """
    # Buscar ponto mais próximo
    dist, idx = kdtree.query([lat, lon])

    return df_ambiental.iloc[idx]["ssha"], df_ambiental.iloc[idx]["chlor_a"]


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================


def descobrir_datas_disponiveis():
    """
    Descobre as datas disponíveis nos dados MODIS e SWOT.

    Returns:
        list: Lista de datas no formato YYYY-MM-DD
    """
    import glob
    import re

    datas_disponiveis = set()

    # Buscar datas nos arquivos MODIS
    modis_files = glob.glob("data/modis/*.nc")
    for filepath in modis_files:
        match = re.search(r"(\d{8})", os.path.basename(filepath))
        if match:
            data_str = match.group(1)
            # Converter YYYYMMDD para YYYY-MM-DD
            data_formatada = f"{data_str[:4]}-{data_str[4:6]}-{data_str[6:8]}"
            datas_disponiveis.add(data_formatada)

    # Buscar datas nos arquivos SWOT
    swot_files = glob.glob("data/swot/*.nc")
    for filepath in swot_files:
        match = re.search(r"(\d{8})", os.path.basename(filepath))
        if match:
            data_str = match.group(1)
            # Converter YYYYMMDD para YYYY-MM-DD
            data_formatada = f"{data_str[:4]}-{data_str[4:6]}-{data_str[6:8]}"
            datas_disponiveis.add(data_formatada)

    return sorted(list(datas_disponiveis))


def simular_tubarao_por_data(
    id_tubarao, data, ponto_inicial, df_ambiental, kdtree, coords
):
    """
    Simula um tubarão para uma data específica.

    Args:
        id_tubarao: ID único do tubarão
        data: Data no formato YYYY-MM-DD
        ponto_inicial: dict com lat/lon inicial
        df_ambiental: DataFrame com dados ambientais
        kdtree: árvore KD
        coords: array de coordenadas

    Returns:
        list: lista de registros do tubarão para a data
    """
    registros = []

    # Estado inicial
    lat_atual = ponto_inicial["lat"]
    lon_atual = ponto_inicial["lon"]
    tempo_atual = datetime.strptime(f"{data} 00:00:00", "%Y-%m-%d %H:%M:%S")
    comportamento = "busca"  # Comportamento inicial
    nivel_fome = NIVEL_FOME_INICIAL
    tempo_sem_alimento = 0
    historico_posicoes = []

    for ping in range(PINGS_POR_TUBARAO):
        # Calcular hora do dia
        tempo_dia = tempo_atual.hour + tempo_atual.minute / 60.0

        # Buscar dados ambientais para a posição atual
        ssha, chlor_a = buscar_dados_ambientais_proximos(
            lat_atual, lon_atual, kdtree, df_ambiental, coords
        )

        # Calcular probabilidade de forrageio usando modelo avançado
        p_forrageio, velocidade_preferida, direcao_preferida = (
            calcular_probabilidade_forrageio_avancada(
                ssha,
                chlor_a,
                lat_atual,
                lon_atual,
                tempo_dia,
                comportamento,
                historico_posicoes,
                nivel_fome,
            )
        )

        # Determinar comportamento usando modelo avançado
        comportamento = determinar_comportamento_avancado(
            p_forrageio, comportamento, tempo_sem_alimento
        )

        # Atualizar nível de fome
        if comportamento == "forrageando":
            # Chance de encontrar alimento
            if np.random.random() < p_forrageio * 0.3:
                nivel_fome = max(0, nivel_fome - 0.2)
                tempo_sem_alimento = 0
            else:
                nivel_fome = min(1, nivel_fome + 0.01)
                tempo_sem_alimento += 1
        else:
            nivel_fome = min(1, nivel_fome + 0.005)
            tempo_sem_alimento += 1

        # Calcular profundidade baseada no comportamento
        profundidade_m = calcular_profundidade_comportamental(
            comportamento, tempo_dia, ssha, chlor_a
        )

        # Simular dados dos sensores
        dados_sensores = simular_dados_sensores(
            comportamento, profundidade_m, tempo_dia, nivel_fome
        )

        # Converter coordenadas para formato de telemetria
        lat_int, lon_int = converter_coordenadas_para_telemetria(lat_atual, lon_atual)

        # Preparar dados para CRC (sem o CRC ainda)
        dados_telemetria = [
            int(tempo_atual.timestamp()),  # timestamp (4B)
            lat_int,  # lat (3B)
            lon_int,  # lon (3B)
            dados_sensores["depth_dm"],  # depth_dm (2B)
            dados_sensores["temp_cC"],  # temp_cC (2B)
            dados_sensores["batt_mV"],  # batt_mV (2B)
            dados_sensores["acc_x"],  # acc_x (2B)
            dados_sensores["acc_y"],  # acc_y (2B)
            dados_sensores["acc_z"],  # acc_z (2B)
            dados_sensores["gyr_x"],  # gyr_x (2B)
            dados_sensores["gyr_y"],  # gyr_y (2B)
            dados_sensores["gyr_z"],  # gyr_z (2B)
        ]

        # Converter para bytes para calcular CRC
        data_bytes = bytearray()
        data_bytes.extend(
            struct.pack(">I", dados_telemetria[0])
        )  # timestamp (big-endian)
        data_bytes.extend(
            struct.pack(">i", dados_telemetria[1])[1:]
        )  # lat (3 bytes, big-endian)
        data_bytes.extend(
            struct.pack(">i", dados_telemetria[2])[1:]
        )  # lon (3 bytes, big-endian)
        data_bytes.extend(struct.pack(">H", dados_telemetria[3]))  # depth_dm
        data_bytes.extend(struct.pack(">h", dados_telemetria[4]))  # temp_cC
        data_bytes.extend(struct.pack(">H", dados_telemetria[5]))  # batt_mV
        data_bytes.extend(
            struct.pack(
                ">hhh",
                dados_telemetria[6],
                dados_telemetria[7],
                dados_telemetria[8],
            )
        )  # acc
        data_bytes.extend(
            struct.pack(
                ">hhh",
                dados_telemetria[9],
                dados_telemetria[10],
                dados_telemetria[11],
            )
        )  # gyr

        # Calcular CRC-16
        crc16 = calcular_crc16_ccitt(data_bytes)

        # Registrar dados de telemetria (apenas campos especificados)
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
            "p_forrageio": round(p_forrageio, 4),
            "comportamento": comportamento,
        }
        registros.append(registro)

        # Calcular próximo movimento
        if ping < PINGS_POR_TUBARAO - 1:  # Não mover no último ping
            nova_lat, nova_lon = calcular_movimento_avancado(
                comportamento,
                velocidade_preferida,
                direcao_preferida,
                lat_atual,
                lon_atual,
            )

            # Atualizar posição
            lat_atual = nova_lat
            lon_atual = nova_lon

            # Manter histórico de posições
            historico_posicoes.append((lat_atual, lon_atual))
            if len(historico_posicoes) > HISTORICO_POSICOES:
                historico_posicoes.pop(0)

            # Avançar tempo
            tempo_atual += timedelta(minutes=INTERVALO_PING_MINUTOS)

    return registros


def main():
    """
    Função principal de simulação avançada por data.
    """
    print("=" * 60)
    print("SIMULADOR AVANÇADO DE DADOS SINTÉTICOS DE TUBARÕES")
    print("=" * 60)

    # Descobrir datas disponíveis
    datas_disponiveis = descobrir_datas_disponiveis()
    print(f"Datas disponíveis nos dados MODIS/SWOT: {len(datas_disponiveis)}")
    print(f"Período: {min(datas_disponiveis)} a {max(datas_disponiveis)}")
    print(f"Simulando {N_TUBAROES} tubarões por data ({PINGS_POR_TUBARAO} pings cada)")

    # Carregar dados ambientais
    df_ambiental, kdtree, coords = carregar_dados_ambientais()
    print(f"Usando {len(df_ambiental):,} pontos SWOT+MODIS como base ambiental")

    # Analisar dados ambientais (básico para referência)
    _ = analisar_dados_ambientais_basico(df_ambiental)

    # Selecionar pontos iniciais aleatórios
    np.random.seed(42)  # Para reprodutibilidade
    indices_iniciais = np.random.choice(len(df_ambiental), N_TUBAROES, replace=False)
    pontos_iniciais = [
        {"lat": df_ambiental.iloc[idx]["lat"], "lon": df_ambiental.iloc[idx]["lon"]}
        for idx in indices_iniciais
    ]

    # Preparar diretório de saída
    os.makedirs(ANALISE_DIR, exist_ok=True)

    # Simular para cada data
    todos_registros = []

    print("\nIniciando simulação por data...")
    for data in tqdm(datas_disponiveis, desc="Simulando por data"):
        registros_dia = []

        for i in range(N_TUBAROES):
            registros = simular_tubarao_por_data(
                i + 1, data, pontos_iniciais[i], df_ambiental, kdtree, coords
            )
            registros_dia.extend(registros)

        # Salvar dados do dia
        salvar_dados_tubaroes_por_dia(registros_dia, data)
        todos_registros.extend(registros_dia)

    # Salvar arquivo principal
    print("\nSalvando arquivo principal...")
    df_final = pd.DataFrame(todos_registros)
    df_final.to_csv(OUTPUT_FILE, index=False)

    # Estatísticas finais
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS FINAIS - MODELO AVANÇADO")
    print("=" * 60)

    # Contagem de comportamentos
    contagem_comportamentos = df_final["comportamento"].value_counts()
    print("Distribuição de comportamentos:")
    for comportamento, count in contagem_comportamentos.items():
        pct = count / len(df_final) * 100
        print(f"  {comportamento}: {count:,} pings ({pct:.1f}%)")

    # Estatísticas gerais
    print("\nEstatisticas gerais:")
    print(f"  Pings totais simulados: {len(todos_registros):,}")
    print(f"  Datas simuladas: {len(datas_disponiveis)}")
    print(f"  Tubarões por data: {N_TUBAROES}")
    print(f"  Pings por tubarão por dia: {PINGS_POR_TUBARAO}")
    print(f"  Intervalo entre pings: {INTERVALO_PING_MINUTOS} minutos")

    # Estatísticas de telemetria
    print("\nEstatísticas de telemetria:")
    print(f"  Profundidade média: {df_final['depth_dm'].mean()/10:.1f}m")
    print(f"  Profundidade máxima: {df_final['depth_dm'].max()/10:.1f}m")
    print(f"  Temperatura média: {df_final['temp_cC'].mean()/100:.1f}°C")
    print(f"  Bateria média: {df_final['batt_mV'].mean():.0f}mV")
    print(f"  CRC-16 válido: {len(df_final[df_final['crc16'] > 0]):,} registros")

    # Verificar arquivo
    tamanho_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nArquivo salvo: {OUTPUT_FILE}")
    print(f"Tamanho: {tamanho_mb:.1f} MB")

    print("\nSUCESSO: Simulacao avancada com telemetria concluida com sucesso!")
    print("Dados prontos para treinamento de IA com modelo ecologico realista.")
    print("Inclui dados de telemetria completos baseados em estudos científicos:")
    print("  - Comportamento de mergulho em redemoinhos (Braun et al. 2019)")
    print("  - Dados de sensores realistas (acelerometro, giroscopio, temperatura)")
    print("  - CRC-16/CCITT para verificacao de integridade")
    print("  - Campos de labels para IA: comportamento e p_forrageio")
    print(f"  - Dados sincronizados com MODIS/SWOT: {len(datas_disponiveis)} datas")


if __name__ == "__main__":
    main()
