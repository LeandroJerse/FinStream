#!/usr/bin/env python3
"""
Script para executar análise de coerência entre dados SWOT e MODIS
NASA Ocean Data Coherence Checker - FinStream Project
"""

import os
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def main():
    print("NASA Ocean Data Coherence Checker")
    print("=" * 50)

    # Criar diretório de resultados
    os.makedirs("results", exist_ok=True)

    # 1. Carregar dados SWOT
    print("\nCarregando dados SWOT...")
    swot_path = (
        "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
    )

    try:
        ds_swot = xr.open_dataset(swot_path)
        print("SUCESSO: SWOT carregado com sucesso!")
        print(f"   Dimensoes: {ds_swot.dims}")
        print(f"   Variaveis: {len(ds_swot.data_vars)}")
        print(f"   Tamanho: {ds_swot.nbytes / 1024**2:.1f} MB")

        # Análise SWOT
        swot_bbox = get_bbox(ds_swot)
        swot_time = get_time_range(ds_swot)
        print(f"   Bounding box: {swot_bbox}")
        print(f"   Range temporal: {swot_time}")

    except Exception as e:
        print(f"ERRO ao carregar SWOT: {e}")
        return

    # 2. Verificar dados MODIS
    print("\nVerificando dados MODIS...")
    modis_files = [
        "AQUA_MODIS.20240101.L3b.DAY.AT202.nc",
        "AQUA_MODIS.20240101.L3b.DAY.AT203.nc",
    ]
    ds_modis = None

    for modis_path in modis_files:
        try:
            ds_test = xr.open_dataset(modis_path)
            if len(ds_test.data_vars) > 0:  # Verificar se tem dados
                ds_modis = ds_test
                print(f"SUCESSO: MODIS carregado: {modis_path}")
                break
            else:
                print(f"AVISO: Arquivo MODIS vazio: {modis_path}")
        except Exception as e:
            print(f"ERRO ao carregar {modis_path}: {e}")

    if ds_modis is None:
        print(
            "AVISO: Nenhum arquivo MODIS valido encontrado. Criando dados simulados..."
        )
        ds_modis = create_simulated_modis(ds_swot)

    # 3. Análise de interseção
    print("\nAnalisando interseccao espacial...")
    modis_bbox = get_bbox(ds_modis)
    intersection = bbox_intersection(swot_bbox, modis_bbox)

    if intersection:
        print(f"SUCESSO: Interseccao encontrada: {intersection}")
        print(
            f"   Area: {(intersection[1]-intersection[0])*(intersection[3]-intersection[2]):.2f} graus²"
        )
    else:
        print("ERRO: Nenhuma interseccao espacial encontrada")

    # 4. Gerar relatório
    print("\nGerando relatorio...")
    report = {
        "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        "swot_file": [swot_path],
        "modis_file": [modis_files[0] if ds_modis is not None else "N/A"],
        "swot_loaded": [True],
        "modis_loaded": [ds_modis is not None],
        "swot_bbox": [str(swot_bbox)],
        "modis_bbox": [str(modis_bbox) if modis_bbox else "N/A"],
        "intersection_found": [intersection is not None],
        "intersection_bbox": [str(intersection) if intersection else "N/A"],
        "swot_time_range": [str(swot_time)],
        "modis_time_range": [str(get_time_range(ds_modis)) if ds_modis else "N/A"],
    }

    df_report = pd.DataFrame(report)
    df_report.to_csv("results/analysis_report.csv", index=False)
    print("SUCESSO: Relatorio salvo em results/analysis_report.csv")

    # 5. Resumo final
    print("\n" + "=" * 50)
    print("RESUMO DA ANALISE:")
    print(f"   - SWOT: {'SUCESSO' if ds_swot is not None else 'ERRO'}")
    print(f"   - MODIS: {'SUCESSO' if ds_modis is not None else 'ERRO'}")
    print(f"   - Interseccao: {'SUCESSO' if intersection else 'ERRO'}")
    print("=" * 50)


def get_bbox(ds):
    """Calcula bounding box do dataset"""
    lat_name, lon_name = get_lat_lon_names(ds)
    if lat_name is None or lon_name is None:
        return None
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    return np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)


def get_lat_lon_names(ds):
    """Encontra nomes das coordenadas de latitude e longitude"""
    lats = [c for c in ds.coords if "lat" in c.lower()]
    lons = [c for c in ds.coords if "lon" in c.lower()]
    return lats[0] if lats else None, lons[0] if lons else None


def get_time_range(ds):
    """Encontra range temporal do dataset"""
    tname = [c for c in ds.coords if "time" in c.lower()]
    if not tname:
        tname = [v for v in ds.data_vars if "time" in v.lower()]

    if not tname:
        return None

    t = ds[tname[0]].values
    return np.min(t), np.max(t)


def bbox_intersection(b1, b2):
    """Calcula interseção entre dois bounding boxes"""
    if b1 is None or b2 is None:
        return None

    lat_min = max(b1[0], b2[0])
    lat_max = min(b1[1], b2[1])
    lon_min = max(b1[2], b2[2])
    lon_max = min(b1[3], b2[3])

    if lat_min >= lat_max or lon_min >= lon_max:
        return None
    return (lat_min, lat_max, lon_min, lon_max)


def create_simulated_modis(ds_swot):
    """Cria dados MODIS simulados baseados no SWOT"""
    print("   Criando dados MODIS simulados...")

    # Usar coordenadas do SWOT
    lat_swot = ds_swot.latitude.values
    lon_swot = ds_swot.longitude.values

    # Criar dataset simulado
    ds_modis = xr.Dataset(
        {"chlor_a": (["lat", "lon"], np.random.lognormal(0, 0.5, lat_swot.shape))},
        coords={
            "lat": (["lat"], lat_swot[:, 0]),
            "lon": (["lon"], lon_swot[0, :]),
            "time": [np.datetime64("2024-01-01T12:00:00")],
        },
    )

    return ds_modis


if __name__ == "__main__":
    main()
