#!/usr/bin/env python3
"""
Análise de Interseção Espacial e Temporal - Dados Reais SWOT x MODIS
NASA Ocean Data Coherence Checker - FinStream Project
Preparação de dados reais para treinamento de neurônio
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

    # Acessar BinList para obter informações espaciais
    if "BinList" in ds_modis.data_vars:
        binlist = ds_modis["BinList"]
        binlist_values = binlist.values

        # Extrair informações dos bins
        if hasattr(binlist_values, "dtype") and "bin_num" in binlist_values.dtype.names:
            bin_nums = binlist_values["bin_num"]

            # Converter números de bin para lat/lon usando algoritmo ISIN (Integerized Sinusoidal)
            lats, lons = converter_bin_para_lat_lon(bin_nums)

            print(f"   Coordenadas extraídas: {len(lats)} pontos")
            print(f"   Range lat: {np.nanmin(lats):.3f} a {np.nanmax(lats):.3f}")
            print(f"   Range lon: {np.nanmin(lons):.3f} a {np.nanmax(lons):.3f}")

            return lats, lons

    print("ERRO: Não foi possível extrair coordenadas do MODIS")
    return None, None


def converter_bin_para_lat_lon(bin_nums):
    """Converte números de bin MODIS para coordenadas lat/lon"""
    # Algoritmo ISIN (Integerized Sinusoidal Grid) da NASA
    # Baseado na documentação da NASA Ocean Color

    # Parâmetros do grid ISIN
    num_rows = 2160  # Número de linhas no grid ISIN
    num_cols = 4320  # Número de colunas no grid ISIN

    lats = []
    lons = []

    for bin_num in bin_nums:
        # Converter bin number para row/col
        row = bin_num // num_cols
        col = bin_num % num_cols

        # Converter row/col para lat/lon
        lat = 90.0 - (row + 0.5) * (180.0 / num_rows)
        lon = (col + 0.5) * (360.0 / num_cols) - 180.0

        lats.append(lat)
        lons.append(lon)

    return np.array(lats), np.array(lons)


def processar_dados_modis_reais(ds_modis, lats, lons):
    """Processa dados MODIS reais com coordenadas reais"""
    print("Processando dados MODIS reais...")

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


def analisar_swot_real(ds_swot):
    """Analisa dados SWOT reais"""
    print("Analisando dados SWOT reais...")

    # Extrair dados SSHA
    if "ssha_karin" in ds_swot.data_vars:
        ssha_data = ds_swot["ssha_karin"].values

        # Filtrar dados válidos
        mask_validos = ~np.isnan(ssha_data)

        # Extrair coordenadas usando o mask
        swot_lats = ds_swot["latitude"].values[mask_validos]
        swot_lons = ds_swot["longitude"].values[mask_validos]

        # Extrair tempo usando o mask
        time_data = ds_swot["time"].values
        if len(time_data.shape) == 1:
            # Tempo é 1D, precisa expandir para match com dados 2D
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


def encontrar_interseccao_espacial(swot_data, modis_data, tolerancia=0.5):
    """Encontra interseção espacial entre dados SWOT e MODIS - VERSÃO OTIMIZADA"""
    print(f"\nEncontrando interseção espacial (tolerância: {tolerancia}°)...")

    # OTIMIZAÇÃO: Amostrar pontos para tornar computável
    max_swot_points = 1000  # Limitar pontos SWOT
    max_modis_points = 5000  # Limitar pontos MODIS

    print(f"   Amostrando {max_swot_points} pontos SWOT de {len(swot_data['lat'])}")
    print(f"   Amostrando {max_modis_points} pontos MODIS de {len(modis_data['lat'])}")

    # CORREÇÃO: Fixar semente para resultados reprodutíveis
    np.random.seed(42)

    # Amostrar pontos SWOT
    swot_indices = np.random.choice(
        len(swot_data["lat"]),
        min(max_swot_points, len(swot_data["lat"])),
        replace=False,
    )

    # Amostrar pontos MODIS
    modis_indices = np.random.choice(
        len(modis_data["lat"]),
        min(max_modis_points, len(modis_data["lat"])),
        replace=False,
    )

    pontos_interseccao = []

    # Para cada ponto SWOT amostrado, procurar pontos MODIS próximos
    for i, swot_idx in enumerate(swot_indices):
        if i % 100 == 0:  # Progress indicator
            print(f"   Processando ponto SWOT {i+1}/{len(swot_indices)}")

        swot_lat = swot_data["lat"][swot_idx]
        swot_lon = swot_data["lon"][swot_idx]

        # Calcular distâncias para pontos MODIS amostrados
        dist_lat = np.abs(modis_data["lat"][modis_indices] - swot_lat)
        dist_lon = np.abs(modis_data["lon"][modis_indices] - swot_lon)

        # Ajustar longitude para distância mínima (considerando -180/180)
        dist_lon = np.minimum(dist_lon, 360 - dist_lon)

        # Encontrar pontos dentro da tolerância
        mask_proximos = (dist_lat <= tolerancia) & (dist_lon <= tolerancia)

        if np.any(mask_proximos):
            # Pegar o ponto MODIS mais próximo
            distancias = np.sqrt(dist_lat**2 + dist_lon**2)
            idx_modis_local = np.argmin(distancias[mask_proximos])
            idx_modis_global = modis_indices[mask_proximos][idx_modis_local]

            ponto = {
                "swot_idx": swot_idx,
                "modis_idx": idx_modis_global,
                "lat": swot_lat,
                "lon": swot_lon,
                "ssha": swot_data["ssha"][swot_idx],
                "chlor_a": modis_data["chlor_a"][idx_modis_global],
                "dist_lat": dist_lat[mask_proximos][idx_modis_local],
                "dist_lon": dist_lon[mask_proximos][idx_modis_local],
                "time": swot_data["time"][swot_idx],
            }
            pontos_interseccao.append(ponto)

    print(f"   Pontos de interseção encontrados: {len(pontos_interseccao)}")

    if pontos_interseccao:
        # Calcular estatísticas
        distancias = [
            p["dist_lat"] ** 2 + p["dist_lon"] ** 2 for p in pontos_interseccao
        ]
        dist_media = np.sqrt(np.mean(distancias))
        dist_max = np.sqrt(np.max(distancias))

        print(f"   Distância média: {dist_media:.3f}°")
        print(f"   Distância máxima: {dist_max:.3f}°")

        # Estatísticas dos dados
        sshas = [p["ssha"] for p in pontos_interseccao]
        chlors = [p["chlor_a"] for p in pontos_interseccao]

        print(
            f"   Range SSHA (interseção): {np.min(sshas):.3f} - {np.max(sshas):.3f} m"
        )
        print(
            f"   Range Clorofila (interseção): {np.min(chlors):.3f} - {np.max(chlors):.3f} mg/m³"
        )

    return pontos_interseccao


def analisar_correlacao_real(pontos_interseccao):
    """Analisa correlação entre SSHA e clorofila usando dados reais"""
    if len(pontos_interseccao) < 10:
        print("Poucos pontos para análise de correlação")
        return None

    print("\nAnalisando correlação SSHA vs Clorofila (dados reais)...")

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
        }
    )

    return df, corr_pearson


def salvar_dados_treinamento(
    df, corr_pearson, arquivo_saida="dados_treinamento_real.csv"
):
    """Salva dados preparados para treinamento"""
    print(f"\nSalvando dados para treinamento: {arquivo_saida}")

    # Adicionar metadados
    df["correlation"] = corr_pearson
    df["data_type"] = "real_intersection"
    df["source"] = "SWOT_MODIS_NASA"
    df["date_processed"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Salvar CSV
    df.to_csv(arquivo_saida, index=False)
    print(f"   Dados salvos: {len(df)} pontos")
    print(f"   Colunas: {list(df.columns)}")

    return arquivo_saida


def gerar_relatorio_final(swot_data, modis_data, pontos_interseccao, corr_pearson):
    """Gera relatório final da análise"""
    print("\n" + "=" * 70)
    print("RELATÓRIO FINAL - ANÁLISE DE INTERSEÇÃO REAL")
    print("=" * 70)

    print(f"DADOS SWOT:")
    print(f"   - Pontos válidos: {len(swot_data['ssha'])}")
    print(
        f"   - Range SSHA: {np.nanmin(swot_data['ssha']):.3f} - {np.nanmax(swot_data['ssha']):.3f} m"
    )
    print(
        f"   - Cobertura espacial: {np.nanmin(swot_data['lat']):.1f}° a {np.nanmax(swot_data['lat']):.1f}° lat"
    )
    print(
        f"   - Cobertura espacial: {np.nanmin(swot_data['lon']):.1f}° a {np.nanmax(swot_data['lon']):.1f}° lon"
    )

    print(f"\nDADOS MODIS:")
    print(f"   - Pontos válidos: {len(modis_data['chlor_a'])}")
    print(
        f"   - Range clorofila: {np.nanmin(modis_data['chlor_a']):.3f} - {np.nanmax(modis_data['chlor_a']):.3f} mg/m³"
    )
    print(
        f"   - Cobertura espacial: {np.nanmin(modis_data['lat']):.1f}° a {np.nanmax(modis_data['lat']):.1f}° lat"
    )
    print(
        f"   - Cobertura espacial: {np.nanmin(modis_data['lon']):.1f}° a {np.nanmax(modis_data['lon']):.1f}° lon"
    )

    print(f"\nINTERSEÇÃO ESPACIAL:")
    print(f"   - Pontos de interseção: {len(pontos_interseccao)}")
    if pontos_interseccao:
        print(f"   - Correlação SSHA vs Clorofila: {corr_pearson:.3f}")
        print(
            f"   - Percentual de cobertura SWOT: {len(pontos_interseccao)/len(swot_data['ssha'])*100:.1f}%"
        )
        print(
            f"   - Percentual de cobertura MODIS: {len(pontos_interseccao)/len(modis_data['chlor_a'])*100:.1f}%"
        )

    print(f"\nAPLICABILIDADE PARA TREINAMENTO:")
    if len(pontos_interseccao) >= 100:
        print(f"   SUCESSO: DADOS SUFICIENTES para treinamento de neuronio")
        print(
            f"   SUCESSO: CORRELACAO {'forte' if abs(corr_pearson) > 0.5 else 'moderada' if abs(corr_pearson) > 0.3 else 'fraca'}"
        )
    else:
        print(
            f"   AVISO: POUCOS DADOS para treinamento ({len(pontos_interseccao)} pontos)"
        )

    print("=" * 70)


def main():
    print("ANÁLISE DE INTERSEÇÃO ESPACIAL - DADOS REAIS SWOT x MODIS")
    print("=" * 70)
    print("Preparação de dados reais para treinamento de neurônio")
    print("=" * 70)

    # Criar diretório de resultados
    os.makedirs("results", exist_ok=True)

    # 1. Carregar dados SWOT
    print("\n1. CARREGANDO DADOS SWOT...")
    swot_path = (
        "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
    )

    try:
        ds_swot = xr.open_dataset(swot_path)
        print("SUCESSO: SWOT carregado")
        swot_data = analisar_swot_real(ds_swot)
        if swot_data is None:
            print("ERRO: Não foi possível processar dados SWOT")
            return
    except Exception as e:
        print(f"ERRO ao carregar SWOT: {e}")
        return

    # 2. Carregar dados MODIS
    print("\n2. CARREGANDO DADOS MODIS...")
    modis_files = [
        "AQUA_MODIS.20240101.L3b.DAY.AT202.nc",
        "AQUA_MODIS.20240101.L3b.DAY.AT203.nc",
    ]

    modis_data = None
    for modis_path in modis_files:
        try:
            print(f"   Tentando: {modis_path}")
            ds_modis = ler_modis_l3b(modis_path)
            if ds_modis is not None:
                lats, lons = extrair_coordenadas_modis(ds_modis)
                if lats is not None and lons is not None:
                    modis_data = processar_dados_modis_reais(ds_modis, lats, lons)
                    if modis_data:
                        print(f"SUCESSO: MODIS processado - {modis_path}")
                        break
        except Exception as e:
            print(f"ERRO ao processar {modis_path}: {e}")

    if modis_data is None:
        print("ERRO: Não foi possível processar dados MODIS")
        return

    # 3. Verificar áreas de cobertura
    print("\n3. VERIFICANDO ÁREAS DE COBERTURA...")
    print(
        f"   SWOT: lat {np.min(swot_data['lat']):.1f}° a {np.max(swot_data['lat']):.1f}°, lon {np.min(swot_data['lon']):.1f}° a {np.max(swot_data['lon']):.1f}°"
    )
    print(
        f"   MODIS: lat {np.min(modis_data['lat']):.1f}° a {np.max(modis_data['lat']):.1f}°, lon {np.min(modis_data['lon']):.1f}° a {np.max(modis_data['lon']):.1f}°"
    )

    # Calcular sobreposição
    lat_overlap = min(np.max(swot_data["lat"]), np.max(modis_data["lat"])) - max(
        np.min(swot_data["lat"]), np.min(modis_data["lat"])
    )
    lon_overlap = min(np.max(swot_data["lon"]), np.max(modis_data["lon"])) - max(
        np.min(swot_data["lon"]), np.min(modis_data["lon"])
    )

    print(f"   Sobreposição latitudinal: {lat_overlap:.1f}°")
    print(f"   Sobreposição longitudinal: {lon_overlap:.1f}°")

    if lat_overlap <= 0 or lon_overlap <= 0:
        print("   AVISO: NENHUMA SOBREPOSICAO ESPACIAL DETECTADA!")
        print("   As areas de cobertura SWOT e MODIS nao se sobrepoem.")
        return
    else:
        print(
            f"   SUCESSO: Sobreposicao detectada: {lat_overlap*lon_overlap:.1f} graus²"
        )

    # 4. Encontrar interseção espacial com tolerância maior
    print("\n4. ENCONTRANDO INTERSEÇÃO ESPACIAL...")
    pontos_interseccao = None

    # Tentar com tolerâncias maiores
    for tolerancia in [0.5, 1.0, 2.0]:
        print(f"\n   Tentando com tolerância: {tolerancia}°")
        pontos_interseccao = encontrar_interseccao_espacial(
            swot_data, modis_data, tolerancia=tolerancia
        )
        if pontos_interseccao:
            print(f"   SUCESSO: Interseccao encontrada com tolerancia {tolerancia}°")
            break

    if not pontos_interseccao:
        print("ERRO: Nenhuma interseção espacial encontrada mesmo com tolerância alta")
        return

    # 5. Analisar correlação
    print("\n5. ANALISANDO CORRELAÇÃO...")
    resultado_corr = analisar_correlacao_real(pontos_interseccao)

    if resultado_corr:
        df, corr_pearson = resultado_corr

        # 6. Salvar dados para treinamento
        print("\n6. SALVANDO DADOS PARA TREINAMENTO...")
        arquivo_saida = salvar_dados_treinamento(df, corr_pearson)

        # 7. Relatório final
        gerar_relatorio_final(swot_data, modis_data, pontos_interseccao, corr_pearson)

        print(f"\nSUCESSO: ANALISE CONCLUIDA COM SUCESSO!")
        print(f"   Arquivo de dados: {arquivo_saida}")
        print(f"   Pronto para treinamento de neuronio!")
    else:
        print("ERRO: Não foi possível analisar correlação")


if __name__ == "__main__":
    main()
