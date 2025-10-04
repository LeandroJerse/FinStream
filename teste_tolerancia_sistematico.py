#!/usr/bin/env python3
"""
Teste Sistemático de Tolerâncias - Análise de Interseção SWOT x MODIS
Verifica se o comportamento com diferentes tolerâncias está correto
"""

import numpy as np
import pandas as pd
import xarray as xr

from analise_intersecao_real import (
    analisar_swot_real,
    extrair_coordenadas_modis,
    ler_modis_l3b,
    processar_dados_modis_reais,
)


def teste_tolerancia_sistematico():
    """Testa diferentes tolerâncias de forma sistemática"""
    print("TESTE SISTEMÁTICO DE TOLERÂNCIAS")
    print("=" * 50)

    # Carregar dados
    print("Carregando dados...")

    # SWOT
    ds_swot = xr.open_dataset(
        "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
    )
    swot_data = analisar_swot_real(ds_swot)

    # MODIS
    ds_modis = ler_modis_l3b("AQUA_MODIS.20240101.L3b.DAY.AT202.nc")
    lats, lons = extrair_coordenadas_modis(ds_modis)
    modis_data = processar_dados_modis_reais(ds_modis, lats, lons)

    print(f"SWOT: {len(swot_data['lat'])} pontos")
    print(f"MODIS: {len(modis_data['lat'])} pontos")

    # Testar tolerâncias
    tolerancias = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    resultados = []

    for tolerancia in tolerancias:
        print(f"\n--- Testando tolerância: {tolerancia}° ---")

        # Fixar semente para reprodutibilidade
        np.random.seed(42)

        # Amostrar pontos (mesmo método do script original)
        max_swot_points = 1000
        max_modis_points = 5000

        swot_indices = np.random.choice(
            len(swot_data["lat"]),
            min(max_swot_points, len(swot_data["lat"])),
            replace=False,
        )

        modis_indices = np.random.choice(
            len(modis_data["lat"]),
            min(max_modis_points, len(modis_data["lat"])),
            replace=False,
        )

        pontos_interseccao = []

        for swot_idx in swot_indices:
            swot_lat = swot_data["lat"][swot_idx]
            swot_lon = swot_data["lon"][swot_idx]

            # Calcular distâncias
            dist_lat = np.abs(modis_data["lat"][modis_indices] - swot_lat)
            dist_lon = np.abs(modis_data["lon"][modis_indices] - swot_lon)
            dist_lon = np.minimum(dist_lon, 360 - dist_lon)

            # Encontrar pontos dentro da tolerância
            mask_proximos = (dist_lat <= tolerancia) & (dist_lon <= tolerancia)

            if np.any(mask_proximos):
                distancias = np.sqrt(dist_lat**2 + dist_lon**2)
                idx_modis_local = np.argmin(distancias[mask_proximos])
                idx_modis_global = modis_indices[mask_proximos][idx_modis_local]

                ponto = {
                    "ssha": swot_data["ssha"][swot_idx],
                    "chlor_a": modis_data["chlor_a"][idx_modis_global],
                    "lat": swot_lat,
                    "lon": swot_lon,
                    "distancia": distancias[mask_proximos][idx_modis_local],
                }
                pontos_interseccao.append(ponto)

        # Calcular estatísticas
        n_pontos = len(pontos_interseccao)

        if n_pontos > 0:
            sshas = [p["ssha"] for p in pontos_interseccao]
            chlors = [p["chlor_a"] for p in pontos_interseccao]
            distancias = [p["distancia"] for p in pontos_interseccao]

            corr = np.corrcoef(sshas, chlors)[0, 1] if len(sshas) > 1 else 0
            dist_media = np.mean(distancias)
            dist_max = np.max(distancias)
        else:
            corr = 0
            dist_media = 0
            dist_max = 0

        resultado = {
            "tolerancia": tolerancia,
            "pontos": n_pontos,
            "correlacao": corr,
            "distancia_media": dist_media,
            "distancia_max": dist_max,
        }

        resultados.append(resultado)

        print(f"   Pontos encontrados: {n_pontos}")
        print(f"   Correlação: {corr:.3f}")
        print(f"   Distância média: {dist_media:.3f}°")
        print(f"   Distância máxima: {dist_max:.3f}°")

    # Criar DataFrame com resultados
    df_resultados = pd.DataFrame(resultados)

    print("\n" + "=" * 50)
    print("RESUMO DOS RESULTADOS:")
    print("=" * 50)
    print(df_resultados.to_string(index=False))

    # Verificar se o comportamento está correto
    print("\nVERIFICAÇÃO DO COMPORTAMENTO:")

    # Os pontos DEVERIAM aumentar com tolerância maior
    pontos_crescentes = all(
        df_resultados.iloc[i]["pontos"] >= df_resultados.iloc[i - 1]["pontos"]
        for i in range(1, len(df_resultados))
    )

    if pontos_crescentes:
        print("SUCESSO: COMPORTAMENTO CORRETO - Pontos aumentam com tolerancia maior")
    else:
        print("ERRO: COMPORTAMENTO INCORRETO - Pontos nao aumentam consistentemente")
        print("   Isso indica problema na lógica de amostragem")

    # Salvar resultados
    df_resultados.to_csv("resultados_tolerancia_sistematico.csv", index=False)
    print(f"\nResultados salvos em: resultados_tolerancia_sistematico.csv")

    return df_resultados


if __name__ == "__main__":
    teste_tolerancia_sistematico()
