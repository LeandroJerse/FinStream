# Conversor de NetCDF para CSV

Script para converter arquivos NetCDF (.nc) da NASA (SWOT e MODIS) para formato CSV.

## 📋 Requisitos

```bash
pip install numpy pandas xarray tqdm
```

## 🚀 Uso

### 1. Converter um arquivo específico

```bash
# Detecta automaticamente se é SWOT ou MODIS
python converter_nc_para_csv.py arquivo.nc

# Especificar nome do arquivo de saída
python converter_nc_para_csv.py arquivo.nc --output meu_arquivo.csv
```

### 2. Converter todos os arquivos SWOT

```bash
# Converte todos os .nc em data/swot/ para CSV
python converter_nc_para_csv.py --all-swot

# Usar diretório diferente
python converter_nc_para_csv.py --all-swot --swot-dir caminho/para/swot
```

### 3. Converter todos os arquivos MODIS

```bash
# Converte todos os .nc em data/modis/ para CSV
python converter_nc_para_csv.py --all-modis

# Usar diretório diferente
python converter_nc_para_csv.py --all-modis --modis-dir caminho/para/modis
```

### 4. Converter tudo de uma vez

```bash
# Converte SWOT + MODIS
python converter_nc_para_csv.py --all
```

## 📊 Formato dos Arquivos CSV

### SWOT (Sea Surface Height)
```csv
lat,lon,ssha
-10.5234,45.2341,0.1234
-10.5235,45.2342,0.1245
...
```

**Colunas:**
- `lat`: Latitude (graus)
- `lon`: Longitude (graus)
- `ssha`: Sea Surface Height Anomaly (metros)

### MODIS (Clorofila-a)
```csv
lat,lon,chlor_a
-10.5234,45.2341,0.025678
-10.5235,45.2342,0.026123
...
```

**Colunas:**
- `lat`: Latitude (graus)
- `lon`: Longitude (graus)
- `chlor_a`: Concentração de clorofila-a (mg/m³)

## 🔍 Detalhes Técnicos

### SWOT
- **Variável extraída:** `ssha_karin`
- **Coordenadas:** `latitude_avg_ssh`, `longitude_avg_ssh`
- **Processamento:** Remove valores NaN, ajusta dimensões se necessário

### MODIS
- **Variável extraída:** `chlor_a`
- **Grupo NetCDF:** `level-3_binned_data`
- **Coordenadas:** Convertidas de bins para lat/lon
- **Processamento:** Calcula média (sum/nobs), remove valores inválidos

## 📁 Estrutura de Saída

Quando usa `--all-swot` ou `--all-modis`, os CSVs são salvos em:

```
data/
├── swot/
│   ├── SWOT_*.nc
│   └── csv/
│       ├── SWOT_*.csv
│       └── ...
└── modis/
    ├── AQUA_MODIS_*.nc
    └── csv/
        ├── AQUA_MODIS_*.csv
        └── ...
```

## ⚙️ Opções Avançadas

```bash
# Ver ajuda completa
python converter_nc_para_csv.py --help

# Especificar diretórios customizados
python converter_nc_para_csv.py --all \
    --swot-dir meus_dados/swot \
    --modis-dir meus_dados/modis
```

## 🐛 Solução de Problemas

### Erro: "Variável 'ssha_karin' não encontrada"
- O arquivo não é um arquivo SWOT válido
- Tente converter como MODIS: use `converter_modis_para_csv()`

### Erro: "Estrutura de dados inesperada"
- O arquivo MODIS pode ter formato diferente
- Verifique se o arquivo tem o grupo `level-3_binned_data`

### Arquivo vazio ou poucos pontos
- Os dados podem ter muitos valores NaN
- Isso é normal para dados oceânicos com cobertura parcial

## 📝 Exemplos Práticos

```bash
# Exemplo 1: Converter SWOT específico
python converter_nc_para_csv.py \
    data/swot/SWOT_L2_LR_SSH_Expert_008_497_20240101T000705_20240101T005833_PIC0_01.nc

# Exemplo 2: Converter MODIS específico
python converter_nc_para_csv.py \
    data/modis/AQUA_MODIS.20240101.L3b.DAY.AT202.nc

# Exemplo 3: Processar tudo em lote
python converter_nc_para_csv.py --all

# Exemplo 4: Converter e renomear
python converter_nc_para_csv.py arquivo.nc --output dados_oceanicos.csv
```

## 📊 Integração com Pipeline

Este script pode ser usado antes do pipeline principal:

```bash
# 1. Converter arquivos NetCDF para CSV
python converter_nc_para_csv.py --all

# 2. (Opcional) Processar os CSVs gerados
# Os CSVs podem ser lidos facilmente com pandas
import pandas as pd
df = pd.read_csv('data/swot/csv/SWOT_arquivo.csv')
```

## 🔗 Links Úteis

- **SWOT:** https://swot.jpl.nasa.gov/data/
- **MODIS Ocean Color:** https://oceancolor.gsfc.nasa.gov/
- **xarray:** https://xarray.pydata.org/
