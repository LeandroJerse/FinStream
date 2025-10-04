#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulador Avançado de Dados Sintéticos de Tubarões
==================================================

Gera dados realistas de movimento de tubarões baseados em modelos ecológicos
avançados, considerando produtividade oceânica, dinâmica de correntes,
ritmos circadianos e comportamento de forrageio ótimo.

Autor: Sistema de Análise Oceanográfica Avançada
Data: 2024-01-01
"""

import os
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit as sigmoid
from tqdm import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURAÇÕES AVANÇADAS
# =============================================================================

# Parâmetros da simulação
N_TUBAROES = 50
PINGS_POR_TUBARAO = 1000
INTERVALO_PING_MINUTOS = 5
DATA_INICIO = "2024-01-01 00:00:00"

# Parâmetros de movimento por comportamento (graus)
PASSO_FORRAGEIO_BASE = 0.003  # ~200m em 5min
PASSO_BUSCA_BASE = 0.008  # ~500m em 5min
PASSO_TRANSITO_BASE = 0.02  # ~1.5km em 5min

# Parâmetros comportamentais
PROB_CONTINUAR_COMPORTAMENTO = 0.4  # Reduzido para mais variabilidade
PROB_MUDAR_COMPORTAMENTO = 0.6
HISTORICO_POSICOES = 10  # Número de posições para calcular velocidade
FADIGA_NUTRICIONAL_BASE = 0.1  # Fadiga nutricional inicial

# Arquivos
DADOS_AMBIENTAIS = "data/dados_treinamento_ia.csv"
OUTPUT_FILE = "data/tubaroes_sinteticos.csv"
ANALISE_DIR = "data/analise_diaria"

# =============================================================================
# FUNÇÕES AUXILIARES AVANÇADAS
# =============================================================================


def carregar_dados_ambientais():
    """
    Carrega e prepara os dados ambientais reais (SWOT + MODIS).

    Returns:
        tuple: (df, kdtree, coords_array)
    """
    print("Carregando dados ambientais reais...")

    # Carregar apenas colunas necessárias para economizar memória
    df = pd.read_csv(DADOS_AMBIENTAIS, usecols=["lat", "lon", "ssha", "chlor_a"])

    # Converter para numérico e remover NaN
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["ssha"] = pd.to_numeric(df["ssha"], errors="coerce")
    df["chlor_a"] = pd.to_numeric(df["chlor_a"], errors="coerce")

    df_clean = df.dropna().reset_index(drop=True)

    print(f"Dados carregados: {len(df_clean):,} pontos válidos")
    print(
        f"SSHA range: {df_clean['ssha'].min():.2f} a " f"{df_clean['ssha'].max():.2f}"
    )
    print(
        f"Chlor_a range: {df_clean['chlor_a'].min():.4f} a "
        f"{df_clean['chlor_a'].max():.4f}"
    )

    # Criar array de coordenadas para KDTree
    coords = np.column_stack([df_clean["lat"], df_clean["lon"]])
    kdtree = cKDTree(coords)

    return df_clean, kdtree, coords


def calcular_probabilidade_forrageio_avancada(
    ssha,
    chlor_a,
    lat,
    lon,
    tempo_dia,
    comportamento_anterior,
    historico_posicoes,
    fadiga_nutricional=0.0,
):
    """
    Calcula probabilidade de forrageio usando modelo ecológico avançado.

    Baseado em:
    - Gradientes de produtividade (clorofila)
    - Dinâmica de correntes (SSHA)
    - Ritmos circadianos
    - Comportamento de forrageio ótimo
    - Fadiga nutricional

    Args:
        ssha: altura da superfície do mar (correntes/fronteiras)
        chlor_a: concentração de clorofila (produtividade)
        lat, lon: posição geográfica
        tempo_dia: hora do dia (0-24)
        comportamento_anterior: comportamento no ping anterior
        historico_posicoes: últimas N posições para calcular velocidade
        fadiga_nutricional: fator de fome (0-1)

    Returns:
        tuple: (p_forrageio, velocidade_preferida, direcao_preferida)
    """

    # 1. PRODUTIVIDADE PRIMÁRIA (clorofila)
    # Tubarões preferem áreas de alta produtividade
    chlor_norm = np.clip(chlor_a, 0.0024, 73.6333)
    chlor_norm = (chlor_norm - 0.0024) / (73.6333 - 0.0024)

    # 2. DINÂMICA OCEÂNICA (SSHA)
    # Alturas positivas = convergência = alta produtividade
    # Alturas negativas = divergência = baixa produtividade
    ssha_norm = np.clip(ssha, -9.26, 261.41)
    ssha_norm = (ssha_norm - (-9.26)) / (261.41 - (-9.26))

    # Fronteiras oceânicas (gradientes de SSHA)
    gradiente_ssha = np.abs(ssha_norm - 0.5) * 2  # 0-1, máximo nas bordas

    # 3. RITMO CIRCADIANO
    # Tubarões são mais ativos ao amanhecer e anoitecer
    if 5 <= tempo_dia <= 8 or 17 <= tempo_dia <= 20:
        fator_circadiano = 1.3  # Maior atividade
    elif 10 <= tempo_dia <= 16:
        fator_circadiano = 0.8  # Menor atividade (meio dia)
    else:
        fator_circadiano = 1.1  # Atividade moderada (noite)

    # 4. COMPORTAMENTO DE FORRAGEIO ÓTIMO
    # Probabilidade baseada em múltiplos fatores

    # Fator produtividade (clorofila alta = mais forrageio)
    fator_produtividade = sigmoid(2.0 * chlor_norm - 0.5)

    # Fator convergência (SSHA positiva = convergência = alimento)
    fator_convergencia = sigmoid(1.5 * ssha_norm - 0.3)

    # Fator fronteira (gradientes = hotspots de vida)
    fator_fronteira = sigmoid(1.2 * gradiente_ssha - 0.2)

    # 5. INÉRCIA COMPORTAMENTAL
    if comportamento_anterior == "forrageando":
        inercia = 0.3  # Tendência a continuar forrageando
    elif comportamento_anterior == "transitando":
        inercia = -0.2  # Tendência a parar de transitar
    else:  # busca
        inercia = 0.0

    # 6. FADIGA NUTRICIONAL (fome)
    # Tubarões famintos têm maior probabilidade de forragear
    fator_fome = sigmoid(2.0 * fadiga_nutricional - 0.5)

    # 7. FATOR TEMPORAL (varia ao longo do dia)
    # Simula mudanças nas condições oceânicas
    ciclo_diario = np.sin(2 * np.pi * tempo_dia / 24) * 0.2

    # 8. CÁLCULO FINAL DA PROBABILIDADE (mais variável)
    p_forrageio = sigmoid(
        0.4 * fator_produtividade  # Muito reduzido
        + 0.3 * fator_convergencia  # Muito reduzido
        + 0.2 * fator_fronteira  # Muito reduzido
        + 0.2 * fator_circadiano  # Muito reduzido
        + 0.6 * fator_fome  # Aumentado (fome é o principal fator)
        + 0.1 * inercia  # Reduzido
        + ciclo_diario  # Variação temporal
        + np.random.normal(0, 0.3)  # Muito mais ruído
    )

    # 9. VELOCIDADE PREFERIDA baseada no comportamento (limiares mais equilibrados)
    if p_forrageio > 0.55:  # Ainda mais reduzido
        # Forrageio: movimentos lentos e erráticos
        velocidade_base = PASSO_FORRAGEIO_BASE
        variacao_velocidade = 0.5
    elif p_forrageio < 0.45:  # Ainda mais aumentado
        # Trânsito: movimentos rápidos e direcionais
        velocidade_base = PASSO_TRANSITO_BASE
        variacao_velocidade = 0.2
    else:
        # Busca: movimentos intermediários
        velocidade_base = PASSO_BUSCA_BASE
        variacao_velocidade = 0.4

    # Ajustar velocidade baseada na produtividade local
    velocidade_preferida = velocidade_base * (
        1 + variacao_velocidade * np.random.uniform(-1, 1)
    )

    # 10. DIREÇÃO PREFERIDA (limiares ajustados)
    # Em forrageio: direção aleatória
    # Em trânsito: direção baseada em gradientes de produtividade
    if p_forrageio > 0.55:  # Ajustado para consistência
        direcao_preferida = np.random.uniform(0, 2 * np.pi)
    elif p_forrageio < 0.45:  # Ajustado para consistência
        # Direção para áreas de maior produtividade
        direcao_preferida = np.random.uniform(0, 2 * np.pi)  # Simplificado
    else:
        # Busca: direção semi-aleatória com tendência
        direcao_preferida = np.random.uniform(0, 2 * np.pi)

    return float(p_forrageio), float(velocidade_preferida), float(direcao_preferida)


def determinar_comportamento_avancado(
    p_forrageio, comportamento_atual, tempo_sem_alimento=0
):
    """
    Determina o próximo comportamento usando modelo avançado.

    Args:
        p_forrageio: probabilidade de forrageio (0-1)
        comportamento_atual: comportamento atual
        tempo_sem_alimento: tempo desde última refeição (pings)

    Returns:
        str: novo comportamento
    """
    # Chance de continuar no mesmo comportamento (com inércia)
    if np.random.random() < PROB_CONTINUAR_COMPORTAMENTO:
        return comportamento_atual

    # Mudança de comportamento baseada em múltiplos fatores

    # Fator tempo sem alimento
    if tempo_sem_alimento > 150:  # Muito faminto (reduzido de 200)
        if p_forrageio > 0.3:  # Reduzido de 0.4
            return "forrageando"
        else:
            return "busca"

    # Comportamento baseado na probabilidade de forrageio (limiares mais equilibrados)
    if p_forrageio > 0.55:  # Ainda mais reduzido
        return "forrageando"
    elif p_forrageio < 0.45:  # Ainda mais aumentado
        return "transitando"
    else:
        return "busca"


def calcular_movimento_avancado(
    comportamento, velocidade_preferida, direcao_preferida, lat_atual, lon_atual
):
    """
    Calcula movimento baseado em comportamento e ambiente.

    Args:
        comportamento: comportamento atual
        velocidade_preferida: velocidade em graus
        direcao_preferida: direção em radianos
        lat_atual, lon_atual: posição atual

    Returns:
        tuple: (nova_lat, nova_lon)
    """
    if comportamento == "forrageando":
        # Movimento errático com pequenos passos
        ruido_direcao = np.random.normal(direcao_preferida, 0.5)
        ruido_velocidade = velocidade_preferida * np.random.uniform(0.3, 1.5)

    elif comportamento == "transitando":
        # Movimento direcional com poucos desvios
        ruido_direcao = np.random.normal(direcao_preferida, 0.1)
        ruido_velocidade = velocidade_preferida * np.random.uniform(0.8, 1.2)

    else:  # busca
        # Movimento intermediário
        ruido_direcao = np.random.normal(direcao_preferida, 0.3)
        ruido_velocidade = velocidade_preferida * np.random.uniform(0.5, 1.3)

    # Converter para deslocamento em lat/lon
    # Aproximação simples para pequenas distâncias
    deslocamento_lat = ruido_velocidade * np.cos(ruido_direcao)
    deslocamento_lon = ruido_velocidade * np.sin(ruido_direcao)

    # Ajustar longitude baseado na latitude (convergência dos meridianos)
    fator_lon = 1.0 / np.cos(np.radians(lat_atual))
    deslocamento_lon *= fator_lon

    nova_lat = lat_atual + deslocamento_lat
    nova_lon = lon_atual + deslocamento_lon

    return nova_lat, nova_lon


def analisar_dados_ambientais_basico(df_ambiental):
    """
    Análise básica dos dados ambientais para referência da IA.

    Args:
        df_ambiental: DataFrame com dados ambientais

    Returns:
        dict: Estatísticas básicas
    """
    print("\nAnalisando dados ambientais para referência da IA...")

    # Criar diretório de análise
    os.makedirs(ANALISE_DIR, exist_ok=True)

    # Estatísticas gerais (apenas para referência)
    stats_gerais = {
        "total_pontos": len(df_ambiental),
        "ssha_media": df_ambiental["ssha"].mean(),
        "ssha_std": df_ambiental["ssha"].std(),
        "chlor_media": df_ambiental["chlor_a"].mean(),
        "chlor_std": df_ambiental["chlor_a"].std(),
        "correlacao_ssha_chlor": df_ambiental["ssha"].corr(df_ambiental["chlor_a"]),
    }

    print(f"Dados ambientais carregados: {stats_gerais['total_pontos']:,} pontos")
    print(f"SSHA médio: {stats_gerais['ssha_media']:.3f}")
    print(f"Chlor_a médio: {stats_gerais['chlor_media']:.4f}")

    return {"stats_gerais": stats_gerais}


def salvar_dados_tubaroes_por_dia(registros_dia, data):
    """
    Salva apenas os dados CSV dos tubarões por dia (para treinamento da IA).

    Args:
        registros_dia: Lista de registros do dia
        data: Data do arquivo
    """
    if not registros_dia:
        return

    df_dia = pd.DataFrame(registros_dia)

    # Salvar apenas dados CSV dos tubarões
    arquivo_csv = f"{ANALISE_DIR}/tubaroes_{data.replace('-', '')}.csv"
    df_dia.to_csv(arquivo_csv, index=False)

    print(f"Dados tubarões salvos: {arquivo_csv}")


def buscar_dados_ambientais_proximos(lat, lon, kdtree, df_ambiental, coords):
    """
    Busca os dados ambientais mais próximos para uma posição.

    Args:
        lat, lon: coordenadas
        kdtree: árvore KD dos dados ambientais
        df_ambiental: DataFrame com dados ambientais
        coords: array de coordenadas

    Returns:
        tuple: (ssha, chlor_a)
    """
    # Buscar ponto mais próximo
    dist, idx = kdtree.query([lat, lon])

    return df_ambiental.iloc[idx]["ssha"], df_ambiental.iloc[idx]["chlor_a"]


def simular_tubarao_avancado(id_tubarao, ponto_inicial, df_ambiental, kdtree, coords):
    """
    Simula um tubarão individual com algoritmo avançado.

    Args:
        id_tubarao: ID único do tubarão
        ponto_inicial: dict com lat/lon inicial
        df_ambiental: DataFrame com dados ambientais
        kdtree: árvore KD
        coords: array de coordenadas

    Returns:
        list: lista de registros do tubarão
    """
    registros = []

    # Estado inicial
    lat_atual = ponto_inicial["lat"]
    lon_atual = ponto_inicial["lon"]
    tempo_atual = datetime.strptime(DATA_INICIO, "%Y-%m-%d %H:%M:%S")
    comportamento = "busca"  # Comportamento inicial
    fadiga_nutricional = FADIGA_NUTRICIONAL_BASE
    tempo_sem_alimento = 0
    historico_posicoes = []

    for ping in range(PINGS_POR_TUBARAO):
        # Calcular hora do dia
        tempo_dia = tempo_atual.hour + tempo_atual.minute / 60.0

        # Buscar dados ambientais para a posição atual
        ssha, chlor_a = buscar_dados_ambientais_proximos(
            lat_atual, lon_atual, kdtree, df_ambiental, coords
        )

        # Calcular probabilidade de forrageio usando modelo avançado
        p_forrageio, velocidade_preferida, direcao_preferida = (
            calcular_probabilidade_forrageio_avancada(
                ssha,
                chlor_a,
                lat_atual,
                lon_atual,
                tempo_dia,
                comportamento,
                historico_posicoes,
                fadiga_nutricional,
            )
        )

        # Determinar comportamento usando modelo avançado
        comportamento = determinar_comportamento_avancado(
            p_forrageio, comportamento, tempo_sem_alimento
        )

        # Atualizar fadiga nutricional
        if comportamento == "forrageando":
            # Chance de encontrar alimento
            if np.random.random() < p_forrageio * 0.3:
                fadiga_nutricional = max(0, fadiga_nutricional - 0.2)
                tempo_sem_alimento = 0
            else:
                fadiga_nutricional = min(1, fadiga_nutricional + 0.01)
                tempo_sem_alimento += 1
        else:
            fadiga_nutricional = min(1, fadiga_nutricional + 0.005)
            tempo_sem_alimento += 1

        # Registrar posição atual
        registro = {
            "id_tubarao": id_tubarao,
            "tempo": tempo_atual.strftime("%Y-%m-%d %H:%M:%S"),
            "lat": round(lat_atual, 6),
            "lon": round(lon_atual, 6),
            "ssha": round(ssha, 4),
            "chlor_a": round(chlor_a, 6),
            "comportamento": comportamento,
            "p_forrageio": round(p_forrageio, 4),
            "velocidade": round(velocidade_preferida * 111000, 2),  # m/min
            "fadiga_nutricional": round(fadiga_nutricional, 3),
        }
        registros.append(registro)

        # Calcular próximo movimento
        if ping < PINGS_POR_TUBARAO - 1:  # Não mover no último ping
            nova_lat, nova_lon = calcular_movimento_avancado(
                comportamento,
                velocidade_preferida,
                direcao_preferida,
                lat_atual,
                lon_atual,
            )

            # Atualizar posição
            lat_atual = nova_lat
            lon_atual = nova_lon

            # Manter histórico de posições
            historico_posicoes.append((lat_atual, lon_atual))
            if len(historico_posicoes) > HISTORICO_POSICOES:
                historico_posicoes.pop(0)

            # Avançar tempo
            tempo_atual += timedelta(minutes=INTERVALO_PING_MINUTOS)

    return registros


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================


def main():
    """
    Função principal de simulação avançada.
    """
    print("=" * 60)
    print("SIMULADOR AVANÇADO DE DADOS SINTÉTICOS DE TUBARÕES")
    print("=" * 60)
    print(
        f"Simulando {N_TUBAROES} tubarões ({PINGS_POR_TUBARAO} pings "
        f"cada, {INTERVALO_PING_MINUTOS} min intervalo)"
    )

    # Carregar dados ambientais
    df_ambiental, kdtree, coords = carregar_dados_ambientais()
    print(f"Usando {len(df_ambiental):,} pontos SWOT+MODIS como base " f"ambiental")

    # Analisar dados ambientais (básico para referência)
    analise_ambiental = analisar_dados_ambientais_basico(df_ambiental)

    # Selecionar pontos iniciais aleatórios
    np.random.seed(42)  # Para reprodutibilidade
    indices_iniciais = np.random.choice(len(df_ambiental), N_TUBAROES, replace=False)
    pontos_iniciais = [
        {"lat": df_ambiental.iloc[idx]["lat"], "lon": df_ambiental.iloc[idx]["lon"]}
        for idx in indices_iniciais
    ]

    # Preparar arquivo de saída
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Cabeçalho do CSV
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(
            "id_tubarao,tempo,lat,lon,ssha,chlor_a,comportamento,"
            "p_forrageio,velocidade,fadiga_nutricional\n"
        )

    # Simular cada tubarão
    todos_registros = []
    correlacoes = []
    registros_por_dia = {}  # Para análise diária

    print("\nIniciando simulação avançada...")
    for i in tqdm(range(N_TUBAROES), desc="Simulando tubarões"):
        registros = simular_tubarao_avancado(
            i + 1, pontos_iniciais[i], df_ambiental, kdtree, coords
        )

        # Salvar incrementalmente
        with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
            for registro in registros:
                f.write(
                    f"{registro['id_tubarao']},{registro['tempo']},"
                    f"{registro['lat']},{registro['lon']},"
                    f"{registro['ssha']},{registro['chlor_a']},"
                    f"{registro['comportamento']},"
                    f"{registro['p_forrageio']},"
                    f"{registro['velocidade']},"
                    f"{registro['fadiga_nutricional']}\n"
                )

        # Coletar dados para estatísticas
        todos_registros.extend(registros)

        # Organizar por dia para análise
        for registro in registros:
            data = registro["tempo"][:10]  # YYYY-MM-DD
            if data not in registros_por_dia:
                registros_por_dia[data] = []
            registros_por_dia[data].append(registro)

        # Calcular correlação local (últimos 100 pings)
        if len(registros) >= 100:
            df_local = pd.DataFrame(registros[-100:])
            corr = df_local["ssha"].corr(df_local["chlor_a"])
            if not np.isnan(corr):
                correlacoes.append(corr)

    # Salvar dados dos tubarões por dia
    print("\nSalvando dados dos tubarões por dia...")
    for data, registros_dia in registros_por_dia.items():
        salvar_dados_tubaroes_por_dia(registros_dia, data)

    # Estatísticas finais
    print("\n" + "=" * 60)
    print("ESTATÍSTICAS FINAIS - MODELO AVANÇADO")
    print("=" * 60)

    # Correlação média
    corr_media = np.mean(correlacoes) if correlacoes else 0
    print(f"Correlação média SSHA-Chlor_a: {corr_media:+.2f}")

    # Contagem de comportamentos
    df_final = pd.DataFrame(todos_registros)
    contagem_comportamentos = df_final["comportamento"].value_counts()
    print("Distribuição de comportamentos:")
    for comportamento, count in contagem_comportamentos.items():
        pct = count / len(df_final) * 100
        print(f"  {comportamento}: {count:,} pings ({pct:.1f}%)")

    # Estatísticas de velocidade
    velocidades = df_final["velocidade"]
    print(f"\nEstatísticas de velocidade:")
    print(f"  Velocidade média: {velocidades.mean():.1f} m/min")
    print(f"  Velocidade máxima: {velocidades.max():.1f} m/min")
    print(f"  Velocidade mínima: {velocidades.min():.1f} m/min")

    # Estatísticas de fadiga nutricional
    fadiga = df_final["fadiga_nutricional"]
    print(f"\nEstatísticas de fadiga nutricional:")
    print(f"  Fadiga média: {fadiga.mean():.3f}")
    print(f"  Fadiga máxima: {fadiga.max():.3f}")

    # Estatísticas gerais
    print(f"\nEstatísticas gerais:")
    print(f"  Pings totais simulados: {len(todos_registros):,}")
    print("  Período simulado: 3 dias por tubarão")
    print(f"  Intervalo entre pings: {INTERVALO_PING_MINUTOS} minutos")

    # Verificar arquivo
    tamanho_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"\nArquivo salvo: {OUTPUT_FILE}")
    print(f"Tamanho: {tamanho_mb:.1f} MB")

    print("\nSUCESSO: Simulação avançada concluída com sucesso!")
    print("Dados prontos para treinamento de IA com modelo ecológico realista.")


if __name__ == "__main__":
    main()
