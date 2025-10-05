# 🔄 Conversor Automático NetCDF → CSV

Script automatizado para converter todos os arquivos NetCDF (.nc) das pastas `data/modis` e `data/swot` para CSV.

## 🚀 Uso Rápido

### Converter tudo automaticamente:

```bash
python converter_nc_automatico.py
```

Pronto! O script vai:
1. ✅ Buscar todos os `.nc` em `data/modis/` → converter para `modisCSV/`
2. ✅ Buscar todos os `.nc` em `data/swot/` → converter para `swotCSV/`
3. ✅ Mostrar progresso em tempo real
4. ✅ Exibir resumo final

## 📁 Estrutura de Pastas

### Antes:
```
FinStream/
├── data/
│   ├── modis/
│   │   ├── AQUA_MODIS.20240101.L3b.DAY.AT202.nc
│   │   ├── AQUA_MODIS.20240102.L3b.DAY.AT202.nc
│   │   └── ...
│   └── swot/
│       ├── SWOT_L2_LR_SSH_Expert_*.nc
│       └── ...
```

### Depois:
```
FinStream/
├── modisCSV/          ← 🆕 NOVA PASTA
│   ├── AQUA_MODIS.20240101.L3b.DAY.AT202.csv
│   ├── AQUA_MODIS.20240102.L3b.DAY.AT202.csv
│   └── ...
├── swotCSV/           ← 🆕 NOVA PASTA
│   ├── SWOT_L2_LR_SSH_Expert_*.csv
│   └── ...
├── data/
│   ├── modis/         ← Arquivos originais preservados
│   └── swot/          ← Arquivos originais preservados
```

## 📊 Formato dos CSV Gerados

### SWOT CSV (swotCSV/)
```csv
lat,lon,ssha
-10.5234,45.2341,0.1234
-10.5235,45.2342,0.1245
...
```

### MODIS CSV (modisCSV/)
```csv
lat,lon,chlor_a
-10.5234,45.2341,0.025678
-10.5235,45.2342,0.026123
...
```

## ⚙️ Personalizar Pastas

Edite o arquivo `converter_nc_automatico.py` nas linhas 24-27:

```python
PASTA_MODIS_ENTRADA = "data/modis"      # Onde estão os .nc MODIS
PASTA_SWOT_ENTRADA = "data/swot"        # Onde estão os .nc SWOT
PASTA_MODIS_SAIDA = "modisCSV"          # Onde salvar CSV MODIS
PASTA_SWOT_SAIDA = "swotCSV"            # Onde salvar CSV SWOT
```

## 📋 Exemplo de Execução

```bash
$ python converter_nc_automatico.py

======================================================================
CONVERSOR AUTOMÁTICO DE NETCDF PARA CSV
======================================================================

📡 PROCESSANDO ARQUIVOS SWOT
----------------------------------------------------------------------
Encontrados 10 arquivos SWOT

Convertendo SWOT: 100%|████████████████████| 10/10 [00:45<00:00,  4.5s/it]

✅ SWOT: 10 convertidos com sucesso
📁 Salvos em: swotCSV/

🛰️  PROCESSANDO ARQUIVOS MODIS
----------------------------------------------------------------------
Encontrados 3 arquivos MODIS

Convertendo MODIS: 100%|██████████████████| 3/3 [00:12<00:00,  4.1s/it]

✅ MODIS: 3 convertidos com sucesso
📁 Salvos em: modisCSV/

======================================================================
RESUMO FINAL
======================================================================
SWOT:  10 sucessos, 0 falhas
MODIS: 3 sucessos, 0 falhas
TOTAL: 13 arquivos CSV gerados

✨ CONVERSÃO CONCLUÍDA!
======================================================================
```

## 🔧 Requisitos

```bash
pip install numpy pandas xarray tqdm
```

## 💡 Dicas

1. **Processar apenas SWOT ou MODIS:** Comente as seções que não quer processar
2. **Verificar arquivos:** Os `.nc` originais **não são alterados**
3. **Re-executar:** Sobrescreve CSVs existentes (pode executar quantas vezes quiser)
4. **Arquivos grandes:** Processamento pode demorar alguns minutos

## ❓ Solução de Problemas

### "Nenhum arquivo .nc encontrado"
- Verifique se os arquivos estão nas pastas corretas
- Confirme que as pastas `data/modis` e `data/swot` existem

### Erros durante conversão
- Alguns arquivos podem estar corrompidos (normal)
- O script continua processando os demais arquivos
- Verifique o resumo final para ver quantos tiveram sucesso

## 🎯 Próximos Passos

Após converter, você pode:

1. **Usar os CSVs diretamente:**
   ```python
   import pandas as pd
   df = pd.read_csv('swotCSV/SWOT_arquivo.csv')
   ```

2. **Processar no pipeline:**
   - Os CSVs podem ser usados em análises
   - Mais fácil de manipular que NetCDF
   - Compatível com Excel, R, Python, etc.

## 📞 Suporte

Problemas? Verifique:
- ✅ Arquivos .nc estão nas pastas corretas
- ✅ Bibliotecas instaladas (`xarray`, `pandas`, etc.)
- ✅ Espaço em disco suficiente para os CSVs
