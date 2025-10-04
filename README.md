# FinStream - Análise de Coerência SWOT x MODIS

Projeto de análise de interseção espacial entre dados SWOT (Sea Surface Height Anomaly) e MODIS (Clorofila-a) da NASA para treinamento de redes neurais.

## 📊 Dados Utilizados

### SWOT (Surface Water Ocean Topography)
- **Arquivo:** `SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc`
- **Fonte:** NASA JPL SWOT Mission
- **Variáveis:** SSHA (Sea Surface Height Anomaly), latitude, longitude, tempo
- **Pontos:** 343.001 pontos válidos

### MODIS (Moderate Resolution Imaging Spectroradiometer)
- **Arquivos:** 
  - `AQUA_MODIS.20240101.L3b.DAY.AT202.nc` (103 MB)
  - `AQUA_MODIS.20240101.L3b.DAY.AT203.nc` (103 MB)
- **Fonte:** NASA Ocean Biology Processing Group
- **Variáveis:** Clorofila-a, latitude, longitude
- **Pontos:** 794.409 pontos válidos

## 🚀 Como Executar

### Pré-requisitos
```bash
pip install numpy pandas xarray scipy matplotlib
```

### Obter Dados
1. Baixe os arquivos .nc da NASA (links no final do README)
2. Coloque-os no diretório raiz do projeto
3. Execute o script principal:

```bash
python analise_ultra_eficiente.py
```

## 📈 Resultados

### Algoritmo Otimizado
- **Método:** cKDTree + processamento em lotes
- **Complexidade:** O(log n) vs O(n²) do método tradicional
- **Tempo:** ~2 minutos para processar todos os dados

### Dados de Treinamento
- **Arquivo:** `dados_treinamento_ultra_eficiente.csv`
- **Pontos:** 119.868 interseções espaciais
- **Correlação:** -0.002 (SSHA vs Clorofila)
- **Tolerância:** 1.0° (excelente precisão espacial)

### Colunas do Dataset
- `ssha`: Sea Surface Height Anomaly (m)
- `chlor_a`: Clorofila-a (mg/m³)
- `lat`: Latitude (graus)
- `lon`: Longitude (graus)
- `time`: Timestamp
- `distancia`: Distância entre pontos SWOT e MODIS (graus)
- `correlation`: Correlação Pearson
- `data_type`: Tipo de dados (real_intersection_ultra_efficient)
- `source`: Fonte (SWOT_MODIS_NASA_ULTRA_EFFICIENT)

## 🔗 Links para Dados

### SWOT
- [NASA SWOT Data Portal](https://swot.jpl.nasa.gov/data/)

### MODIS
- [NASA Ocean Biology Processing Group](https://oceancolor.gsfc.nasa.gov/)

## 📁 Estrutura do Projeto

```
FinStream/
├── analise_ultra_eficiente.py          # Script principal
├── dados_treinamento_ultra_eficiente.csv # Dados para treinamento
├── results/                            # Relatórios e visualizações
├── *.nc                               # Arquivos de dados (não versionados)
└── README.md                          # Este arquivo
```

## 🎯 Aplicação

Os dados processados estão prontos para:
- Treinamento de redes neurais
- Análise de correlação oceanográfica
- Estudos de coerência espacial SWOT-MODIS
- Pesquisa em oceanografia por satélite

## ⚠️ Nota Importante

Os arquivos .nc (dados originais) não são versionados devido ao tamanho (>100MB). Baixe-os separadamente dos links fornecidos.

## 📊 Estatísticas Finais

- **Dados SWOT utilizados:** 343.001 pontos (100%)
- **Dados MODIS utilizados:** 794.409 pontos (100%)
- **Pontos de interseção:** 119.868
- **Cobertura temporal:** 2024-01-01 (51 minutos)
- **Cobertura espacial:** Global (lat: -71° a 77°, lon: 24.9° a 119.3°)

---

**Projeto FinStream - NASA Ocean Data Coherence Checker**
