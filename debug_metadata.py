#!/usr/bin/env python3
"""
Script para debugar metadados
"""

import os
import pickle


def debug_metadata():
    """Debug dos metadados salvos"""

    # Verificar metadados SWOT
    swot_cache = "tmp_cache/swot_meta.pkl"
    if os.path.exists(swot_cache):
        with open(swot_cache, "rb") as f:
            swot_meta = pickle.load(f)

        print("=== METADADOS SWOT ===")
        for meta in swot_meta:
            print(f"Arquivo: {meta['file_name']}")
            print(f"Tipo: {meta['type']}")
            if "time_start" in meta:
                print(f"Tempo início: {meta['time_start']}")
            print()

    # Verificar metadados MODIS
    modis_cache = "tmp_cache/modis_meta.pkl"
    if os.path.exists(modis_cache):
        with open(modis_cache, "rb") as f:
            modis_meta = pickle.load(f)

        print("=== METADADOS MODIS ===")
        for meta in modis_meta:
            print(f"Arquivo: {meta['file_name']}")
            print(f"Tipo: {meta['type']}")
            if "date" in meta:
                print(f"Data: {meta['date']}")
            print()


if __name__ == "__main__":
    debug_metadata()
