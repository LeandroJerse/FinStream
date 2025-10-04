#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste simples para o notebook de coerência SWOT x MODIS
"""

import os
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

# Criar diretório de resultados
os.makedirs("results", exist_ok=True)

print("OK - Bibliotecas importadas com sucesso!")

# Carregar SWOT
swot_path = "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
print("Loading SWOT:", swot_path)

try:
    ds_swot = xr.open_dataset(swot_path)
    print("OK - SWOT dataset carregado com sucesso!")
    print(f"   Dimensoes: {ds_swot.dims}")
    print(f"   Variaveis: {len(ds_swot.data_vars)}")
    print(f"   Coordenadas: {list(ds_swot.coords.keys())}")
    print(f"   Tamanho: {ds_swot.nbytes / 1024**2:.1f} MB")
except Exception as e:
    print(f"ERRO - Erro ao carregar SWOT: {e}")
    ds_swot = None

# Simular MODIS se SWOT estiver disponível
if ds_swot is not None:
    print("\n=== SIMULANDO DADOS MODIS PARA TESTE ===")

    lat_swot = ds_swot.latitude.values
    lon_swot = ds_swot.longitude.values

    # Criar dataset MODIS simulado
    ds_modis = xr.Dataset(
        {"chlor_a": (["lat", "lon"], np.random.lognormal(0, 0.5, lat_swot.shape))},
        coords={
            "lat": (["lat"], lat_swot[:, 0]),
            "lon": (["lon"], lon_swot[0, :]),
            "time": [np.datetime64("2024-01-01T12:00:00")],
        },
    )

    print("OK - Dataset MODIS simulado criado!")
    print(f"   Dimensoes: {ds_modis.dims}")
    print(f"   Variaveis: {list(ds_modis.data_vars.keys())}")
    print(f"   Coordenadas: {list(ds_modis.coords.keys())}")


# Funções auxiliares
def get_bbox(ds):
    lat_name = [c for c in ds.coords if "lat" in c.lower()][0]
    lon_name = [c for c in ds.coords if "lon" in c.lower()][0]
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    return np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)


def bbox_intersection(b1, b2):
    lat_min = max(b1[0], b2[0])
    lat_max = min(b1[1], b2[1])
    lon_min = max(b1[2], b2[2])
    lon_max = min(b1[3], b2[3])
    if lat_min >= lat_max or lon_min >= lon_max:
        return None
    return (lat_min, lat_max, lon_min, lon_max)


# Análise dos dados
if ds_swot is not None and ds_modis is not None:
    print("\n=== ANALISE DOS DADOS ===")

    swot_bbox = get_bbox(ds_swot)
    modis_bbox = get_bbox(ds_modis)

    print("SWOT BBOX:", swot_bbox)
    print("MODIS BBOX:", modis_bbox)

    # Verificar interseção
    intersection = bbox_intersection(swot_bbox, modis_bbox)

    if intersection:
        print("OK - Interseccao espacial encontrada:", intersection)
        print(
            f"   Area de overlap: {intersection[1]-intersection[0]:.2f} graus lat x {intersection[3]-intersection[2]:.2f} graus lon"
        )
    else:
        print("AVISO - Nenhuma interseccao espacial encontrada")

# Criar relatório
report = {
    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "swot_loaded": [ds_swot is not None],
    "modis_loaded": [ds_modis is not None if "ds_modis" in locals() else False],
    "swot_bbox": [str(swot_bbox) if "swot_bbox" in locals() else "N/A"],
    "modis_bbox": [str(modis_bbox) if "modis_bbox" in locals() else "N/A"],
    "intersection_found": [
        intersection is not None if "intersection" in locals() else False
    ],
}

df_report = pd.DataFrame(report)
df_report.to_csv("results/analysis_report.csv", index=False)

print("\n=== RELATORIO FINAL ===")
print(f"   • SWOT carregado: {'OK' if ds_swot is not None else 'ERRO'}")
print(
    f"   • MODIS carregado: {'OK' if 'ds_modis' in locals() and ds_modis is not None else 'ERRO'}"
)
print(
    f"   • Interseccao espacial: {'OK' if 'intersection' in locals() and intersection else 'NAO'}"
)
print(f"\nOK - Relatorio salvo em results/analysis_report.csv")

print("\nPROXIMOS PASSOS:")
print("   1. Baixar dados MODIS reais da NASA")
print("   2. Verificar overlap temporal e espacial")
print("   3. Expandir analise para correlacao entre variaveis")

print("\nSUCESSO - Teste concluido!")

