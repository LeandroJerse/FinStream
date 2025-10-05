#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Monitoramento de Tubarão em Tempo Real
================================================

Sistema que monitora dados de telemetria de tubarão a cada minuto,
une com dados ambientais (SWOT + MODIS) e envia para IA para predição
de comportamento e forrageio.

Autor: FinShark Project - TAG System
Data: 2024
"""

import os
import time
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
import tensorflow as tf
import xarray as xr
from scipy.spatial import cKDTree

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES
# =============================================================================

# Caminhos dos dados
TUBARAO_CSV = (
    "C:/Users/leand/Documents/ufu/Nasa/FinShark/FinStream/"
    "TAG/Data/data_tag_fake/tubarao_tag_simulado.csv"
)
SWOT_DIR = (
    "C:/Users/leand/Documents/ufu/Nasa/FinShark/FinStream/" "TAG/Data/swot_for_loop"
)
MODIS_DIR = (
    "C:/Users/leand/Documents/ufu/Nasa/FinShark/FinStream/" "TAG/Data/modis_for_loop"
)

# Modelo de IA
MODEL_PATH = (
    "C:/Users/leand/Documents/ufu/Nasa/FinShark/FinStream/"
    "TAG/IA/Model/tubarao_comportamento_model.h5"
)
SCALER_PATH = (
    "C:/Users/leand/Documents/ufu/Nasa/FinShark/FinStream/" "TAG/IA/Model/scaler.pkl"
)

# API
API_URL = "https://fb457da07468.ngrok-free.app/api/RastreamentoTubaroes/v1"

# Configurações do sistema
INTERVALO_MINUTOS = 1  # Intervalo entre processamentos
TOLERANCIA_ESPACIAL = 1.0  # Tolerância espacial em graus

# =============================================================================
# FUNÇÕES AUXILIARES
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

            # Converter longitude de 0-360 para -180 a 180
            lons = np.where(lons > 180, lons - 360, lons)

            # Filtrar dados válidos
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
    Carrega dados ambientais (SWOT + MODIS) para uma data específica.
    """
    data_str = data.replace("-", "")
    swot_files = []
    for root, dirs, files in os.walk(SWOT_DIR):
        for file in files:
            if data_str in file and file.endswith(".nc"):
                swot_files.append(os.path.join(root, file))

    modis_files = []
    for root, dirs, files in os.walk(MODIS_DIR):
        for file in files:
            if data_str in file and file.endswith(".nc"):
                modis_files.append(os.path.join(root, file))

    if not swot_files or not modis_files:
        return None, None, None, None, None, None

    # Carregar dados SWOT
    swot_lats, swot_lons, swot_ssha = carregar_dados_swot(swot_files)

    # Carregar dados MODIS
    modis_lats, modis_lons, modis_chlor = carregar_dados_modis(modis_files)

    if len(swot_lats) == 0 or len(modis_lats) == 0:
        return None, None, None, None, None, None

    return swot_lats, swot_lons, swot_ssha, modis_lats, modis_lons, modis_chlor


def buscar_dados_ambientais_proximos(lat, lon, swot_data, modis_data):
    """
    Busca os dados ambientais mais próximos para uma posição.
    """
    swot_lats, swot_lons, swot_ssha = swot_data
    modis_lats, modis_lons, modis_chlor = modis_data

    ssha_ambiente = np.nan
    chlor_a_ambiente = np.nan

    # Buscar SWOT mais próximo
    if len(swot_lats) > 0:
        swot_kdtree = cKDTree(np.column_stack([swot_lats, swot_lons]))
        query_point = np.array([[lat, lon]])
        dist_swot, idx_swot = swot_kdtree.query(query_point, k=1)
        if dist_swot[0] < TOLERANCIA_ESPACIAL:
            ssha_ambiente = swot_ssha[idx_swot[0]]

    # Buscar MODIS mais próximo
    if len(modis_lats) > 0:
        modis_kdtree = cKDTree(np.column_stack([modis_lats, modis_lons]))
        query_point = np.array([[lat, lon]])
        dist_modis, idx_modis = modis_kdtree.query(query_point, k=1)
        if dist_modis[0] < TOLERANCIA_ESPACIAL:
            chlor_a_ambiente = modis_chlor[idx_modis[0]]

    return ssha_ambiente, chlor_a_ambiente


def carregar_modelo_ia():
    """
    Carrega o modelo de IA e o scaler.
    """
    try:
        # Carregar modelo
        custom_objects = {"mse": tf.keras.metrics.MeanSquaredError()}
        model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)

        # Carregar scaler
        scaler = joblib.load(SCALER_PATH)

        print("✅ Modelo de IA carregado com sucesso!")
        return model, scaler
    except Exception as e:
        print(f"❌ Erro ao carregar modelo de IA: {e}")
        return None, None


def fazer_predicao_ia(model, scaler, dados_tubarao):
    """
    Faz predição usando o modelo de IA.
    """
    try:
        # Preparar dados de entrada
        acc_total = np.sqrt(
            dados_tubarao["acc_x"] ** 2
            + dados_tubarao["acc_y"] ** 2
            + dados_tubarao["acc_z"] ** 2
        )

        input_data = np.array(
            [
                [
                    dados_tubarao["timestamp"],
                    dados_tubarao["lat"],
                    dados_tubarao["lon"],
                    dados_tubarao["depth_dm"],
                    dados_tubarao["temp_cC"],
                    dados_tubarao["ssha_ambiente"],
                    dados_tubarao["chlor_a_ambiente"],
                    acc_total,
                ]
            ]
        )

        # Normalizar dados
        input_scaled = scaler.transform(input_data)

        # Fazer predição
        prediction = model.predict(input_scaled, verbose=0)

        # Processar saída
        behavior_probs = prediction[0][0]  # Primeira saída (comportamento)
        forage_prob = prediction[1][0][0]  # Segunda saída (forrageio)

        # Resultados
        behavior_labels = ["busca", "forrageando", "transitando"]
        predicted_behavior = behavior_labels[np.argmax(behavior_probs)]

        return {
            "comportamento": predicted_behavior,
            "probabilidades_comportamento": {
                label: float(prob)
                for label, prob in zip(behavior_labels, behavior_probs)
            },
            "p_forrageio": float(forage_prob),
        }
    except Exception as e:
        print(f"❌ Erro na predição da IA: {e}")
        return None


def enviar_dados_api(dados_completos):
    """
    Envia dados para a API.
    """
    try:

        headers = {
            "Content-Type": "application/json",
            "ngrok-skip-browser-warning": "true",
        }

        response = requests.post(
            API_URL, json=dados_completos, headers=headers, timeout=10
        )

        if response.status_code == 200:
            print("✅ Dados enviados com sucesso para API!")
            return True
        else:
            print(f"❌ Erro ao enviar dados: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erro na comunicação com API: {e}")
        return False


# =============================================================================
# SISTEMA PRINCIPAL
# =============================================================================


class SistemaMonitoramentoTubarao:
    """
    Sistema de monitoramento de tubarão em tempo real.
    """

    def __init__(self):
        self.df_tubarao = None
        self.model = None
        self.scaler = None
        self.swot_data = None
        self.modis_data = None
        self.indice_atual = 0
        self.ultima_data_processada = None

    def inicializar(self):
        """
        Inicializa o sistema.
        """
        print("🚀 Inicializando Sistema de Monitoramento de Tubarão...")

        # Carregar dados do tubarão
        print("📊 Carregando dados do tubarão...")
        try:
            self.df_tubarao = pd.read_csv(TUBARAO_CSV)
            print(f"✅ {len(self.df_tubarao)} registros de tubarão carregados")
        except Exception as e:
            print(f"❌ Erro ao carregar dados do tubarão: {e}")
            return False

        # Carregar modelo de IA
        print("🤖 Carregando modelo de IA...")
        self.model, self.scaler = carregar_modelo_ia()
        if self.model is None or self.scaler is None:
            return False

        # Carregar dados ambientais
        print("🌊 Carregando dados ambientais...")
        if not self.carregar_dados_ambientais():
            return False

        print("✅ Sistema inicializado com sucesso!")
        return True

    def carregar_dados_ambientais(self):
        """
        Carrega dados ambientais para a data atual.
        """
        try:
            # Usar data do primeiro registro do tubarão
            primeiro_timestamp = self.df_tubarao.iloc[0]["timestamp"]
            data = datetime.fromtimestamp(primeiro_timestamp).strftime("%Y-%m-%d")

            print(f"📅 Carregando dados ambientais para {data}...")

            swot_lats, swot_lons, swot_ssha, modis_lats, modis_lons, modis_chlor = (
                carregar_dados_ambientais_por_data(data)
            )

            if swot_lats is None:
                print("❌ Nenhum dado ambiental encontrado!")
                return False

            self.swot_data = (swot_lats, swot_lons, swot_ssha)
            self.modis_data = (modis_lats, modis_lons, modis_chlor)

            print(f"✅ Dados ambientais carregados:")
            print(f"   - SWOT: {len(swot_lats):,} pontos")
            print(f"   - MODIS: {len(modis_lats):,} pontos")

            return True
        except Exception as e:
            print(f"❌ Erro ao carregar dados ambientais: {e}")
            return False

    def processar_proximo_registro(self):
        """
        Processa o próximo registro do tubarão.
        """
        if self.indice_atual >= len(self.df_tubarao):
            print("📋 Todos os registros foram processados. Reiniciando...")
            self.indice_atual = 0

        # Obter registro atual
        registro = self.df_tubarao.iloc[self.indice_atual]

        # Converter coordenadas de telemetria para graus decimais
        lat_tubarao = registro["lat"] / 10000.0
        lon_tubarao = registro["lon"] / 10000.0

        # Buscar dados ambientais próximos
        ssha_ambiente, chlor_a_ambiente = buscar_dados_ambientais_proximos(
            lat_tubarao, lon_tubarao, self.swot_data, self.modis_data
        )

        # Preparar dados de entrada
        dados_inputs = {
            "id": int(registro["id"]),
            "timestamp": int(registro["timestamp"]),
            "lat": int(registro["lat"]),
            "lon": int(registro["lon"]),
            "depth_dm": int(registro["depth_dm"]),
            "temp_cC": int(registro["temp_cC"]),
            "batt_mV": int(registro["batt_mV"]),
            "acc_x": int(registro["acc_x"]),
            "acc_y": int(registro["acc_y"]),
            "acc_z": int(registro["acc_z"]),
            "gyr_x": int(registro["gyr_x"]),
            "gyr_y": int(registro["gyr_y"]),
            "gyr_z": int(registro["gyr_z"]),
            "crc16": int(registro["crc16"]),
            "ssha_ambiente": (
                float(ssha_ambiente) if not np.isnan(ssha_ambiente) else None
            ),
            "chlor_a_ambiente": (
                float(chlor_a_ambiente) if not np.isnan(chlor_a_ambiente) else None
            ),
        }

        # Fazer predição com IA
        predicao = fazer_predicao_ia(self.model, self.scaler, dados_inputs)

        # Estruturar dados no formato correto
        dados_completos = [
            {
                "inputs": dados_inputs,
                "outputs": predicao if predicao else {},
            }
        ]

        # Enviar para API
        sucesso = enviar_dados_api(dados_completos)

        # Log do processamento
        timestamp_str = datetime.fromtimestamp(registro["timestamp"]).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        print(
            f"📡 [{timestamp_str}] Registro {self.indice_atual + 1}/{len(self.df_tubarao)}"
        )
        print(f"   Posição: {lat_tubarao:.4f}, {lon_tubarao:.4f}")
        print(
            f"   SSHA: {ssha_ambiente:.2f}"
            if not np.isnan(ssha_ambiente)
            else "   SSHA: N/A"
        )
        print(
            f"   Chlor_a: {chlor_a_ambiente:.4f}"
            if not np.isnan(chlor_a_ambiente)
            else "   Chlor_a: N/A"
        )
        if predicao:
            print(
                f"   IA: {predicao['comportamento']} (p_forrageio: {predicao['p_forrageio']:.3f})"
            )
        print(f"   API: {'✅' if sucesso else '❌'}")

        self.indice_atual += 1
        return sucesso

    def executar(self):
        """
        Executa o sistema de monitoramento.
        """
        print("🔄 Iniciando monitoramento contínuo...")
        print(f"⏱️  Intervalo: {INTERVALO_MINUTOS} minuto(s)")
        print("🛑 Pressione Ctrl+C para parar")

        try:
            while True:
                inicio = time.time()

                # Processar próximo registro
                self.processar_proximo_registro()

                # Aguardar próximo ciclo
                tempo_processamento = time.time() - inicio
                tempo_espera = max(0, INTERVALO_MINUTOS * 60 - tempo_processamento)

                if tempo_espera > 0:
                    print(f"⏳ Aguardando {tempo_espera:.1f}s para próximo ciclo...\n")
                    time.sleep(tempo_espera)
                else:
                    print("⚠️  Processamento demorou mais que o intervalo!\n")

        except KeyboardInterrupt:
            print("\n🛑 Sistema interrompido pelo usuário.")
        except Exception as e:
            print(f"\n❌ Erro no sistema: {e}")


def main():
    """
    Função principal.
    """
    print("=" * 60)
    print("🦈 SISTEMA DE MONITORAMENTO DE TUBARÃO EM TEMPO REAL")
    print("=" * 60)

    # Criar e inicializar sistema
    sistema = SistemaMonitoramentoTubarao()

    if not sistema.inicializar():
        print("❌ Falha na inicialização do sistema!")
        return 1

    # Executar sistema
    sistema.executar()

    return 0


if __name__ == "__main__":
    exit(main())
