#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para remover colunas p_forrageio e comportamento de dados unificados.
Converte dados de treinamento para formato de inferência (sem labels de IA).

Autor: FinShark Project
Data: 2024
"""

import os
from pathlib import Path

import pandas as pd


def remover_colunas_ia(input_file, output_file=None):
    """
    Remove as colunas p_forrageio e comportamento de um arquivo CSV.

    Args:
        input_file (str): Caminho para o arquivo CSV de entrada
        output_file (str, optional): Caminho para o arquivo CSV de saída.
                                   Se None, será gerado automaticamente.

    Returns:
        str: Caminho do arquivo de saída
    """

    print("=" * 60)
    print("REMOVENDO COLUNAS DE IA (p_forrageio e comportamento)")
    print("=" * 60)

    # Verificar se o arquivo de entrada existe
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")

    print(f"Arquivo de entrada: {input_file}")

    # Carregar dados
    print("Carregando dados...")
    df = pd.read_csv(input_file)
    print(f"Total de registros carregados: {len(df):,}")
    print(f"Colunas originais: {list(df.columns)}")

    # Verificar se as colunas existem
    colunas_para_remover = ["p_forrageio", "comportamento"]
    colunas_existentes = [col for col in colunas_para_remover if col in df.columns]

    if not colunas_existentes:
        print("AVISO: Nenhuma das colunas p_forrageio ou comportamento foi encontrada.")
        print("Nenhuma alteração será feita.")
        return input_file

    print(f"Colunas a serem removidas: {colunas_existentes}")

    # Remover colunas
    df_inferencia = df.drop(columns=colunas_existentes)

    print(f"Colunas após remoção: {list(df_inferencia.columns)}")
    print(f"Registros após remoção: {len(df_inferencia):,}")

    # Gerar nome do arquivo de saída se não especificado
    if output_file is None:
        # Salvar no diretório data/IA/IA_TREINADA
        output_file = "data/IA/IA_TREINADA/dados_unificados_final_inferencia.csv"

    # Criar diretório de saída se necessário
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Salvar arquivo
    print(f"Salvando arquivo de inferência: {output_file}")
    df_inferencia.to_csv(output_file, index=False)

    # Estatísticas finais
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS FINAIS")
    print("=" * 60)
    print(f"Arquivo de entrada: {input_file}")
    print(f"Arquivo de saída: {output_file}")
    print(f"Registros processados: {len(df_inferencia):,}")
    print(f"Colunas removidas: {len(colunas_existentes)}")
    print(f"Colunas finais: {len(df_inferencia.columns)}")

    # Verificar dados de telemetria
    if "depth_dm" in df_inferencia.columns:
        print(f"\nDADOS DE TELEMETRIA:")
        print(f"   Profundidade média: {df_inferencia['depth_dm'].mean()/10:.1f}m")
        print(f"   Profundidade máxima: {df_inferencia['depth_dm'].max()/10:.1f}m")
        print(f"   Temperatura média: {df_inferencia['temp_cC'].mean()/100:.1f}°C")
        print(f"   Bateria média: {df_inferencia['batt_mV'].mean():.0f}mV")
        print(f"   CRC-16 válidos: {df_inferencia['crc16'].notna().sum():,} registros")

    # Verificar dados ambientais
    if "ssha_ambiente" in df_inferencia.columns:
        ssha_validos = df_inferencia["ssha_ambiente"].notna().sum()
        print(f"\nDADOS AMBIENTAIS:")
        print(
            f"   SSHA válidos: {ssha_validos:,} ({ssha_validos/len(df_inferencia)*100:.1f}%)"
        )

    if "chlor_a_ambiente" in df_inferencia.columns:
        chlor_validos = df_inferencia["chlor_a_ambiente"].notna().sum()
        print(
            f"   Chlor_a válidos: {chlor_validos:,} ({chlor_validos/len(df_inferencia)*100:.1f}%)"
        )

    print("\nSUCESSO: Arquivo de inferência criado!")
    return str(output_file)


def main():
    """
    Função principal - sempre usa dados_unificados_final.csv como entrada.
    """
    # Arquivo de entrada fixo
    input_file = "data/dados_unificados_final.csv"

    print("=" * 60)
    print("CONVERSOR AUTOMÁTICO PARA DADOS DE INFERÊNCIA")
    print("=" * 60)
    print(f"Arquivo de entrada fixo: {input_file}")
    print("Remove colunas p_forrageio e comportamento automaticamente")
    print("=" * 60)

    try:
        output_file = remover_colunas_ia(input_file, None)
        print(f"\nArquivo de inferência salvo em: {output_file}")

    except Exception as e:
        print(f"\nERRO: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
