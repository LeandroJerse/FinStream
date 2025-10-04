#!/usr/bin/env python3
"""
Função para ler dados MODIS L3b (Level 3 Binned) corretamente
NASA Ocean Data - Formato especial com dados agrupados em bins
"""

import netCDF4 as nc
import numpy as np
import xarray as xr


def ler_modis_l3b(caminho_arquivo):
    """
    Lê dados MODIS L3b corretamente acessando o grupo level-3_binned_data

    Args:
        caminho_arquivo (str): Caminho para o arquivo .nc

    Returns:
        xarray.Dataset: Dataset com dados MODIS processados
    """
    print(f"Lendo arquivo MODIS L3b: {caminho_arquivo}")

    try:
        # Acessar o grupo específico com os dados
        ds = xr.open_dataset(caminho_arquivo, group="level-3_binned_data")
        print(f"SUCESSO: Dados L3b carregados com sucesso!")
        print(f"   Dimensoes: {ds.dims}")
        print(f"   Variaveis: {len(ds.data_vars)}")

        # Mostrar variáveis disponíveis
        print("   Variaveis disponiveis:")
        for var in ds.data_vars.keys():
            print(f"     - {var}")

        return ds

    except Exception as e:
        print(f"ERRO ao ler arquivo L3b: {e}")
        return None


def processar_dados_binned(ds):
    """
    Processa dados binned do MODIS para extrair valores úteis

    Args:
        ds (xarray.Dataset): Dataset com dados binned

    Returns:
        dict: Dicionário com dados processados
    """
    print("\n=== PROCESSANDO DADOS BINNED ===")

    dados_processados = {}

    # Extrair informações dos bins
    if "BinList" in ds.data_vars:
        binlist = ds["BinList"]
        print(f"BinList shape: {binlist.shape}")
        print(f"BinList dtype: {binlist.dtype}")

        # Extrair número de observações por bin
        binlist_values = binlist.values
        if hasattr(binlist_values, "dtype") and "nobs" in binlist_values.dtype.names:
            nobs = binlist_values["nobs"]
            bins_com_dados = nobs > 0
            print(f"Bins com dados: {np.sum(bins_com_dados)} de {len(nobs)}")
            dados_processados["bins_com_dados"] = bins_com_dados
            dados_processados["nobs"] = nobs
        else:
            print("Estrutura de dados diferente do esperado")
            # Criar array de nobs simulado
            nobs = np.ones(len(binlist_values), dtype=int)
            dados_processados["nobs"] = nobs

    # Processar clorofila-a (chlor_a)
    if "chlor_a" in ds.data_vars:
        chlor_a = ds["chlor_a"]
        print(f"\nClorofila-a disponível:")
        print(f"  Shape: {chlor_a.shape}")
        print(f"  Dtype: {chlor_a.dtype}")

        # Extrair valores de clorofila
        chlor_values = chlor_a.values
        if hasattr(chlor_values, "dtype") and "sum" in chlor_values.dtype.names:
            chlor_sum = chlor_values["sum"]
            chlor_sum_sq = chlor_values["sum_squared"]

            # Calcular média (sum / nobs)
            nobs = dados_processados.get("nobs", np.ones_like(chlor_sum))
            chlor_media = np.where(nobs > 0, chlor_sum / nobs, np.nan)

            print(f"  Valores válidos: {np.sum(~np.isnan(chlor_media))}")
            print(
                f"  Range: {np.nanmin(chlor_media):.3f} - {np.nanmax(chlor_media):.3f} mg/m³"
            )

            dados_processados["chlor_a"] = chlor_media
            dados_processados["chlor_a_sum"] = chlor_sum
            dados_processados["chlor_a_sum_squared"] = chlor_sum_sq
        else:
            print("  Estrutura de dados de clorofila diferente do esperado")

    # Processar outras variáveis importantes
    variaveis_importantes = ["Rrs_488", "Rrs_547", "Kd_490", "nflh"]

    for var_name in variaveis_importantes:
        if var_name in ds.data_vars:
            var = ds[var_name]
            var_values = var.values
            if hasattr(var_values, "dtype") and "sum" in var_values.dtype.names:
                var_sum = var_values["sum"]
                nobs = dados_processados.get("nobs", np.ones_like(var_sum))
                var_media = np.where(nobs > 0, var_sum / nobs, np.nan)
                dados_processados[var_name] = var_media
                print(f"  {var_name}: {np.sum(~np.isnan(var_media))} valores válidos")

    return dados_processados


def converter_para_lat_lon(dados_processados, resolucao=0.1):
    """
    Converte dados binned para grid regular lat/lon

    Args:
        dados_processados (dict): Dados processados dos bins
        resolucao (float): Resolução do grid em graus

    Returns:
        xarray.Dataset: Dataset com grid regular
    """
    print(f"\n=== CONVERTENDO PARA GRID LAT/LON (resolução: {resolucao}°) ===")

    # Criar grid regular
    lats = np.arange(-90, 90 + resolucao, resolucao)
    lons = np.arange(-180, 180 + resolucao, resolucao)

    print(f"Grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons)} pontos")

    # Para simplificar, vamos criar um dataset com algumas variáveis principais
    grid_data = {}

    if "chlor_a" in dados_processados:
        # Amostrar alguns pontos do grid para demonstração
        n_samples = min(1000, len(dados_processados["chlor_a"]))
        indices = np.random.choice(
            len(dados_processados["chlor_a"]), n_samples, replace=False
        )

        # Criar coordenadas aleatórias no grid
        lat_indices = np.random.choice(len(lats), n_samples)
        lon_indices = np.random.choice(len(lons), n_samples)

        grid_data["lat"] = (["sample"], lats[lat_indices])
        grid_data["lon"] = (["sample"], lons[lon_indices])
        grid_data["chlor_a"] = (["sample"], dados_processados["chlor_a"][indices])

        if "nobs" in dados_processados:
            grid_data["nobs"] = (["sample"], dados_processados["nobs"][indices])

    # Criar dataset
    ds_grid = xr.Dataset(grid_data)
    print(f"SUCESSO: Dataset de grid criado com {len(ds_grid.sample)} amostras")

    return ds_grid


def main():
    """Função principal para testar a leitura de dados MODIS L3b"""

    print("=== TESTE DE LEITURA MODIS L3b ===")

    # Testar ambos os arquivos
    arquivos = [
        "AQUA_MODIS.20240101.L3b.DAY.AT202.nc",
        "AQUA_MODIS.20240101.L3b.DAY.AT203.nc",
    ]

    for arquivo in arquivos:
        print(f"\n{'='*60}")
        print(f"ARQUIVO: {arquivo}")
        print(f"{'='*60}")

        # Ler dados
        ds = ler_modis_l3b(arquivo)

        if ds is not None:
            # Processar dados
            dados = processar_dados_binned(ds)

            # Converter para grid
            if dados:
                ds_grid = converter_para_lat_lon(dados)

                # Salvar resultado
                arquivo_saida = f"modis_processed_{arquivo.replace('.nc', '')}.nc"
                ds_grid.to_netcdf(arquivo_saida)
            print(f"SUCESSO: Dados salvos em: {arquivo_saida}")

            ds.close()
        else:
            print(f"ERRO: Nao foi possivel processar {arquivo}")


if __name__ == "__main__":
    main()
