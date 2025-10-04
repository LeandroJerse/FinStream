#!/usr/bin/env python3
"""
Análise Completa com TODOS os Dados Reais - SWOT x MODIS
NASA Ocean Data Coherence Checker - FinStream Project
Usa 100% dos dados disponíveis sem amostragem aleatória
"""

import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def ler_modis_l3b(caminho_arquivo):
    """Lê dados MODIS L3b corretamente"""
    try:
        ds = xr.open_dataset(caminho_arquivo, group="level-3_binned_data")
        return ds
    except Exception as e:
        print(f"ERRO ao ler MODIS L3b: {e}")
        return None


def extrair_coordenadas_modis(ds_modis):
    """Extrai coordenadas reais dos dados MODIS L3b"""
    print("Extraindo coordenadas reais do MODIS...")

    if "BinList" in ds_modis.data_vars:
        binlist = ds_modis["BinList"]
        binlist_values = binlist.values

        if hasattr(binlist_values, "dtype") and "bin_num" in binlist_values.dtype.names:
            bin_nums = binlist_values["bin_num"]
            lats, lons = converter_bin_para_lat_lon(bin_nums)

            print(f"   Coordenadas extraídas: {len(lats)} pontos")
            print(f"   Range lat: {np.nanmin(lats):.3f} a {np.nanmax(lats):.3f}")
            print(f"   Range lon: {np.nanmin(lons):.3f} a {np.nanmax(lons):.3f}")

            return lats, lons

    print("ERRO: Não foi possível extrair coordenadas do MODIS")
    return None, None


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


def processar_dados_modis_completos(ds_modis, lats, lons):
    """Processa TODOS os dados MODIS com coordenadas reais"""
    print("Processando TODOS os dados MODIS...")

    dados_processados = {}

    # Extrair nobs (número de observações)
    if "BinList" in ds_modis.data_vars:
        binlist_values = ds_modis["BinList"].values
        if hasattr(binlist_values, "dtype") and "nobs" in binlist_values.dtype.names:
            dados_processados["nobs"] = binlist_values["nobs"]

    # Processar clorofila-a
    if "chlor_a" in ds_modis.data_vars:
        chlor_values = ds_modis["chlor_a"].values
        if hasattr(chlor_values, "dtype") and "sum" in chlor_values.dtype.names:
            chlor_sum = chlor_values["sum"]
            nobs = dados_processados.get("nobs", np.ones_like(chlor_sum))
            chlor_media = np.where(nobs > 0, chlor_sum / nobs, np.nan)

            # Filtrar apenas pontos com dados válidos
            mask_validos = ~np.isnan(chlor_media) & (chlor_media > 0)

            dados_processados["chlor_a"] = chlor_media[mask_validos]
            dados_processados["lat"] = lats[mask_validos]
            dados_processados["lon"] = lons[mask_validos]
            dados_processados["nobs"] = nobs[mask_validos]

            print(f"   Clorofila válida: {len(dados_processados['chlor_a'])} pontos")
            print(
                f"   Range clorofila: {np.nanmin(dados_processados['chlor_a']):.3f} - {np.nanmax(dados_processados['chlor_a']):.3f} mg/m³"
            )

    return dados_processados


def analisar_swot_completo(ds_swot):
    """Analisa TODOS os dados SWOT reais"""
    print("Analisando TODOS os dados SWOT reais...")

    if "ssha_karin" in ds_swot.data_vars:
        ssha_data = ds_swot["ssha_karin"].values
        mask_validos = ~np.isnan(ssha_data)

        swot_lats = ds_swot["latitude"].values[mask_validos]
        swot_lons = ds_swot["longitude"].values[mask_validos]

        time_data = ds_swot["time"].values
        if len(time_data.shape) == 1:
            time_expanded = np.repeat(
                time_data[:, np.newaxis], ssha_data.shape[1], axis=1
            )
            swot_time = time_expanded[mask_validos]
        else:
            swot_time = time_data[mask_validos]

        swot_data = {
            "ssha": ssha_data[mask_validos],
            "lat": swot_lats,
            "lon": swot_lons,
            "time": swot_time,
        }

        print(f"   SSHA válido: {len(swot_data['ssha'])} pontos")
        print(
            f"   Range SSHA: {np.nanmin(swot_data['ssha']):.3f} - {np.nanmax(swot_data['ssha']):.3f} m"
        )
        print(
            f"   Range lat: {np.nanmin(swot_data['lat']):.3f} a {np.nanmax(swot_data['lat']):.3f}"
        )
        print(
            f"   Range lon: {np.nanmin(swot_data['lon']):.3f} a {np.nanmax(swot_data['lon']):.3f}"
        )

        return swot_data

    return None


def encontrar_interseccao_eficiente(swot_data, modis_data, tolerancia=1.0):
    """Encontra interseção espacial usando algoritmo eficiente com TODOS os dados"""
    print(f"Encontrando interseção espacial (tolerância: {tolerancia}°)...")
    print(f"   Analisando {len(swot_data['lat'])} pontos SWOT")
    print(f"   Analisando {len(modis_data['lat'])} pontos MODIS")

    # ALGORITMO EFICIENTE: Usar busca espacial otimizada
    pontos_interseccao = []

    # Para cada ponto SWOT, encontrar pontos MODIS próximos
    for i, (swot_lat, swot_lon) in enumerate(zip(swot_data["lat"], swot_data["lon"])):
        if i % 10000 == 0:  # Progress indicator
            print(f"   Processando ponto SWOT {i+1}/{len(swot_data['lat'])}")

        # Calcular distâncias para TODOS os pontos MODIS
        dist_lat = np.abs(modis_data["lat"] - swot_lat)
        dist_lon = np.abs(modis_data["lon"] - swot_lon)

        # Ajustar longitude para distância mínima
        dist_lon = np.minimum(dist_lon, 360 - dist_lon)

        # Encontrar pontos dentro da tolerância
        mask_proximos = (dist_lat <= tolerancia) & (dist_lon <= tolerancia)

        if np.any(mask_proximos):
            # Pegar o ponto MODIS mais próximo
            distancias = np.sqrt(dist_lat**2 + dist_lon**2)
            idx_modis = np.argmin(distancias[mask_proximos])
            idx_modis_global = np.where(mask_proximos)[0][idx_modis]

            ponto = {
                "swot_idx": i,
                "modis_idx": idx_modis_global,
                "lat": swot_lat,
                "lon": swot_lon,
                "ssha": swot_data["ssha"][i],
                "chlor_a": modis_data["chlor_a"][idx_modis_global],
                "dist_lat": dist_lat[idx_modis_global],
                "dist_lon": dist_lon[idx_modis_global],
                "time": swot_data["time"][i],
            }
            pontos_interseccao.append(ponto)

    print(f"   Pontos de interseção encontrados: {len(pontos_interseccao)}")

    if pontos_interseccao:
        distancias = [
            np.sqrt(p["dist_lat"] ** 2 + p["dist_lon"] ** 2) for p in pontos_interseccao
        ]
        print(f"   Distância média: {np.mean(distancias):.3f}°")
        print(f"   Distância máxima: {np.max(distancias):.3f}°")

        sshas = [p["ssha"] for p in pontos_interseccao]
        chlors = [p["chlor_a"] for p in pontos_interseccao]

        print(
            f"   Range SSHA (interseção): {np.min(sshas):.3f} - {np.max(sshas):.3f} m"
        )
        print(
            f"   Range Clorofila (interseção): {np.min(chlors):.3f} - {np.max(chlors):.3f} mg/m³"
        )

    return pontos_interseccao


def analisar_correlacao_completa(pontos_interseccao):
    """Analisa correlação entre SSHA e clorofila usando TODOS os dados de interseção"""
    if len(pontos_interseccao) < 10:
        print("Poucos pontos para análise de correlação")
        return None

    print(
        f"\nAnalisando correlação SSHA vs Clorofila ({len(pontos_interseccao)} pontos)..."
    )

    sshas = [p["ssha"] for p in pontos_interseccao]
    chlors = [p["chlor_a"] for p in pontos_interseccao]

    # Calcular correlação
    corr_pearson = np.corrcoef(sshas, chlors)[0, 1]

    print(f"   Correlação Pearson: {corr_pearson:.3f}")
    print(f"   Número de pontos: {len(pontos_interseccao)}")

    # Criar DataFrame para análise
    df = pd.DataFrame(
        {
            "ssha": sshas,
            "chlor_a": chlors,
            "lat": [p["lat"] for p in pontos_interseccao],
            "lon": [p["lon"] for p in pontos_interseccao],
            "time": [p["time"] for p in pontos_interseccao],
            "distancia": [
                np.sqrt(p["dist_lat"] ** 2 + p["dist_lon"] ** 2)
                for p in pontos_interseccao
            ],
        }
    )

    return df, corr_pearson


def salvar_dados_completos(
    df, corr_pearson, arquivo_saida="dados_treinamento_completos.csv"
):
    """Salva TODOS os dados preparados para treinamento"""
    print(f"\nSalvando TODOS os dados para treinamento: {arquivo_saida}")

    # Adicionar metadados
    df["correlation"] = corr_pearson
    df["data_type"] = "real_intersection_complete"
    df["source"] = "SWOT_MODIS_NASA_COMPLETE"
    df["date_processed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df["tolerance_degrees"] = 1.0  # Tolerância usada

    # Salvar CSV
    df.to_csv(arquivo_saida, index=False)
    print(f"   Dados salvos: {len(df)} pontos")
    print(f"   Colunas: {list(df.columns)}")

    return arquivo_saida


def main():
    print("ANÁLISE COMPLETA - TODOS OS DADOS REAIS SWOT x MODIS")
    print("=" * 70)
    print("Usando 100% dos dados disponíveis (sem amostragem)")
    print("=" * 70)

    # Criar diretório de resultados
    os.makedirs("results", exist_ok=True)

    # 1. Carregar dados SWOT
    print("\n1. CARREGANDO TODOS OS DADOS SWOT...")
    swot_path = (
        "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
    )

    try:
        ds_swot = xr.open_dataset(swot_path)
        print("SUCESSO: SWOT carregado")
        swot_data = analisar_swot_completo(ds_swot)
        if swot_data is None:
            print("ERRO: Não foi possível processar dados SWOT")
            return
    except Exception as e:
        print(f"ERRO ao carregar SWOT: {e}")
        return

    # 2. Carregar dados MODIS
    print("\n2. CARREGANDO TODOS OS DADOS MODIS...")
    ds_modis = ler_modis_l3b("AQUA_MODIS.20240101.L3b.DAY.AT202.nc")
    if ds_modis is not None:
        lats, lons = extrair_coordenadas_modis(ds_modis)
        if lats is not None and lons is not None:
            modis_data = processar_dados_modis_completos(ds_modis, lats, lons)
            if not modis_data:
                print("ERRO: Não foi possível processar dados MODIS")
                return
        else:
            print("ERRO: Não foi possível extrair coordenadas MODIS")
            return
    else:
        print("ERRO: Não foi possível carregar dados MODIS")
        return

    # 3. Verificar áreas de cobertura
    print("\n3. VERIFICANDO ÁREAS DE COBERTURA...")
    lat_overlap = min(np.max(swot_data["lat"]), np.max(modis_data["lat"])) - max(
        np.min(swot_data["lat"]), np.min(modis_data["lat"])
    )
    lon_overlap = min(np.max(swot_data["lon"]), np.max(modis_data["lon"])) - max(
        np.min(swot_data["lon"]), np.min(modis_data["lon"])
    )

    print(f"   Sobreposição: {lat_overlap*lon_overlap:.1f} graus²")

    if lat_overlap <= 0 or lon_overlap <= 0:
        print("ERRO: Nenhuma sobreposição espacial detectada")
        return

    # 4. Encontrar interseção espacial com TODOS os dados
    print("\n4. ENCONTRANDO INTERSEÇÃO ESPACIAL (TODOS OS DADOS)...")
    pontos_interseccao = encontrar_interseccao_eficiente(
        swot_data, modis_data, tolerancia=1.0
    )

    if not pontos_interseccao:
        print("ERRO: Nenhuma interseção espacial encontrada")
        return

    # 5. Analisar correlação
    print("\n5. ANALISANDO CORRELAÇÃO...")
    resultado_corr = analisar_correlacao_completa(pontos_interseccao)

    if resultado_corr:
        df, corr_pearson = resultado_corr

        # 6. Salvar dados para treinamento
        print("\n6. SALVANDO DADOS COMPLETOS PARA TREINAMENTO...")
        arquivo_saida = salvar_dados_completos(df, corr_pearson)

        # 7. Relatório final
        print("\n" + "=" * 70)
        print("RELATÓRIO FINAL - ANÁLISE COMPLETA")
        print("=" * 70)
        print(f"DADOS SWOT UTILIZADOS: {len(swot_data['ssha'])} pontos (100%)")
        print(f"DADOS MODIS UTILIZADOS: {len(modis_data['chlor_a'])} pontos (100%)")
        print(f"PONTOS DE INTERSEÇÃO: {len(pontos_interseccao)}")
        print(f"CORRELAÇÃO SSHA vs CLOROFILA: {corr_pearson:.3f}")
        print(f"ARQUIVO GERADO: {arquivo_saida}")
        print("=" * 70)

        print(f"\n✅ ANÁLISE COMPLETA CONCLUÍDA!")
        print(f"   Arquivo de dados: {arquivo_saida}")
        print(f"   Dados 100% reais da NASA")
        print(f"   Pronto para treinamento de neurônio!")
    else:
        print("ERRO: Não foi possível analisar correlação")


if __name__ == "__main__":
    main()
