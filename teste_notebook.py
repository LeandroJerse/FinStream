#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de teste para o notebook de coerência SWOT x MODIS
"""

import os
from datetime import datetime

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# === 1. Imports ===
import xarray as xr
from pandas import to_datetime

# Criar diretório de resultados
os.makedirs("results", exist_ok=True)

print("OK - Bibliotecas importadas com sucesso!")

# === 2. Load datasets ===
swot_path = "SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc"
modis_path = "data/AQUA_MODIS.20240101.L3b.DAY.AT202.nc"

print("Loading SWOT:", swot_path)
try:
    ds_swot = xr.open_dataset(swot_path)
    print("OK - SWOT dataset carregado com sucesso!")
    print(f"   Dimensões: {ds_swot.dims}")
    print(f"   Variáveis: {len(ds_swot.data_vars)}")
except Exception as e:
    print(f"ERRO - Erro ao carregar SWOT: {e}")
    ds_swot = None

print("\nLoading MODIS:", modis_path)
try:
    ds_modis = xr.open_dataset(modis_path)
    print("✅ MODIS dataset carregado com sucesso!")
    print(f"   Dimensões: {ds_modis.dims}")
    print(f"   Variáveis: {len(ds_modis.data_vars)}")
except Exception as e:
    print(f"❌ Erro ao carregar MODIS: {e}")
    print(
        "   Nota: Este arquivo não existe ainda. Vamos simular dados MODIS para teste."
    )
    ds_modis = None

# === 3. Simulação de dados MODIS para teste ===
if ds_modis is None and ds_swot is not None:
    print("\n=== SIMULANDO DADOS MODIS PARA TESTE ===")

    # Usar as mesmas coordenadas do SWOT para simular overlap
    lat_swot = ds_swot.latitude.values
    lon_swot = ds_swot.longitude.values

    # Criar dataset MODIS simulado
    ds_modis = xr.Dataset(
        {"chlor_a": (["lat", "lon"], np.random.lognormal(0, 0.5, lat_swot.shape))},
        coords={
            "lat": (["lat"], lat_swot[:, 0]),  # Usar primeira coluna de latitude
            "lon": (["lon"], lon_swot[0, :]),  # Usar primeira linha de longitude
            "time": [np.datetime64("2024-01-01T12:00:00")],
        },
    )

    print("✅ Dataset MODIS simulado criado!")
    print(f"   Dimensões: {ds_modis.dims}")
    print(f"   Variáveis: {list(ds_modis.data_vars.keys())}")


# === 4. Helper functions ===
def get_lat_lon_names(ds):
    """Encontra nomes das coordenadas de latitude e longitude"""
    lats = [c for c in ds.coords if "lat" in c.lower()]
    lons = [c for c in ds.coords if "lon" in c.lower()]
    return lats[0] if lats else None, lons[0] if lons else None


def get_bbox(ds):
    """Calcula bounding box do dataset"""
    lat_name, lon_name = get_lat_lon_names(ds)
    if lat_name is None or lon_name is None:
        return None
    lats = ds[lat_name].values
    lons = ds[lon_name].values
    return np.nanmin(lats), np.nanmax(lats), np.nanmin(lons), np.nanmax(lons)


def get_time_range(ds):
    """Encontra range temporal do dataset"""
    # Procura primeiro nas coordenadas
    tname = [c for c in ds.coords if "time" in c.lower()]
    if not tname:
        # Se não encontrar nas coordenadas, procura nas variáveis de dados
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


print("✅ Funções auxiliares definidas!")

# === 5. Análise dos dados ===
if ds_swot is not None:
    print("\n=== ANÁLISE SWOT ===")
    swot_bbox = get_bbox(ds_swot)
    swot_time = get_time_range(ds_swot)

    print("SWOT BBOX:", swot_bbox)
    print("SWOT Time range:", swot_time)
    print("SWOT Coordinates:", list(ds_swot.coords.keys()))
    print("SWOT Data variables:", list(ds_swot.data_vars.keys())[:5], "...")
    print(f"Dimensões SWOT: {ds_swot.dims}")
    print(f"Tamanho do dataset: {ds_swot.nbytes / 1024**2:.1f} MB")
else:
    print("❌ Dataset SWOT não disponível")
    swot_bbox = None
    swot_time = None

if ds_modis is not None:
    print("\n=== ANÁLISE MODIS ===")
    modis_bbox = get_bbox(ds_modis)
    modis_time = get_time_range(ds_modis)

    print("MODIS BBOX:", modis_bbox)
    print("MODIS Time range:", modis_time)
    print("MODIS Coordinates:", list(ds_modis.coords.keys()))
    print("MODIS Data variables:", list(ds_modis.data_vars.keys()))
else:
    print("❌ Dataset MODIS não disponível")
    modis_bbox = None
    modis_time = None

# === 6. Verificação de interseção espacial ===
print("\n=== VERIFICAÇÃO DE INTERSEÇÃO ===")

if swot_bbox is not None and modis_bbox is not None:
    intersection = bbox_intersection(swot_bbox, modis_bbox)

    if intersection:
        print("✅ Interseção espacial encontrada:", intersection)
        print(
            f"   Área de overlap: {intersection[1]-intersection[0]:.2f}° lat × {intersection[3]-intersection[2]:.2f}° lon"
        )
    else:
        print("❌ Nenhuma interseção espacial encontrada")
        print("   Os datasets não se sobrepõem geograficamente")
else:
    print("❌ Não é possível verificar interseção - dados insuficientes")
    intersection = None

# === 7. Relatório final ===
print("\n=== RELATÓRIO FINAL ===")

# Criar relatório
report = {
    "timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    "swot_file": [swot_path if ds_swot is not None else "N/A"],
    "modis_file": [modis_path if ds_modis is not None else "N/A"],
    "swot_loaded": [ds_swot is not None],
    "modis_loaded": [ds_modis is not None],
    "swot_bbox": [str(swot_bbox) if swot_bbox else "N/A"],
    "modis_bbox": [str(modis_bbox) if modis_bbox else "N/A"],
    "intersection_found": [intersection is not None],
    "intersection_bbox": [str(intersection) if intersection else "N/A"],
    "swot_time_range": [str(swot_time) if swot_time else "N/A"],
    "modis_time_range": [str(modis_time) if modis_time else "N/A"],
}

# Salvar relatório
df_report = pd.DataFrame(report)
df_report.to_csv("results/analysis_report.csv", index=False)

print("📊 RESUMO DOS RESULTADOS:")
print(f"   • SWOT carregado: {'✅' if ds_swot is not None else '❌'}")
print(f"   • MODIS carregado: {'✅' if ds_modis is not None else '❌'}")
print(f"   • Interseção espacial: {'✅' if intersection else '❌'}")

print(f"\n✅ Relatório salvo em results/analysis_report.csv")

# Mostrar próximos passos
print(f"\n📋 PRÓXIMOS PASSOS:")
if ds_modis is None:
    print("   1. Baixar dados MODIS reais da NASA")
    print("   2. Verificar se há overlap temporal e espacial")
if intersection is None:
    print("   3. Ajustar área de estudo para ter overlap")
else:
    print("   4. Expandir análise para correlação entre variáveis")
    print("   5. Criar visualizações dos dados")

print("\n🎉 Teste concluído com sucesso!")
