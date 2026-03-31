# Projeto 2 — Classificação de Notícias por Compressão

**Disciplina:** Introdução à Teoria da Informação (ITI 2025.2)

**Instituição:** UFPB / CI

**Professor:** Leonardo Vidal Batista

**Alunos:** Gustavo da Rocha, Kelvin Soares Oliveira, Rivando Bezerra Cavalcanti Neto

---

## Escopo

Classificação de notícias em 4 categorias (World, Sports, Business, Sci/Tech) utilizando métodos baseados em compressão de dados, sem técnicas de Machine Learning ou NLP tradicionais.

Dois compressores são utilizados como classificadores:

- **PPM-C** (Prediction by Partial Matching com exclusão) — modelo estatístico que estima probabilidades por contexto
- **LZW** (Lempel-Ziv-Welch) — modelo baseado em dicionário/trie

### Princípio

Um compressor treinado nos textos de uma classe comprime melhor textos daquela mesma classe. Para classificar um texto novo, calculamos o comprimento de código (entropia cruzada) sob cada modelo de classe e atribuímos a classe com menor comprimento.

---

## Dataset

**AG News** — dataset padrão de classificação de notícias em inglês.

| Arquivo     | Textos  | Classes | Textos/classe |
|-------------|---------|---------|---------------|
| `train.csv` | 120.000 | 4       | 30.000        |
| `test.csv`  | 7.600   | 4       | 1.900         |

Classes: 1 = World, 2 = Sports, 3 = Business, 4 = Sci/Tech

---

## Estrutura dos arquivos

```
Projeto2/
  classify.py                 # Classificador (PPM-C e LZW)
  experiments.py              # Script de experimentos e gráficos
  train.csv                   # Dados de treino (AG News)
  test.csv                    # Dados de teste (AG News)
  experiment_results.csv      # Resultados dos experimentos
  experiment_results.png      # Gráficos dos resultados
  main.py                     # Compressor PPM-C do Projeto 1 (referência)
```

---

## Como usar

### Requisitos

```bash
pip install matplotlib python-docx
```

Python 3.10+ recomendado. Nenhuma dependência de ML (scikit-learn, torch, etc.).

### Classificação simples

```bash
# PPM-C com kmax=5 (todos os dados de treino)
python classify.py ppm --kmax 5

# PPM-C com subconjunto de treino (mais rápido)
python classify.py ppm --kmax 5 --max-train 5000

# LZW
python classify.py lzw

# Salvar modelo treinado para reusar
python classify.py ppm --kmax 5 --save-model modelo_ppm5.pkl

# Carregar modelo salvo (pula o treino)
python classify.py ppm --load-model modelo_ppm5.pkl
```

### Parâmetros do classify.py

| Parâmetro       | Default     | Descrição                                |
|-----------------|-------------|------------------------------------------|
| `method`        | (obrigatório) | `ppm` ou `lzw`                         |
| `--kmax`        | 5           | Ordem máxima do PPM                      |
| `--rho`         | 1           | Parâmetro de escape do PPM-C             |
| `--max-train`   | todos       | Limite de textos de treino por classe    |
| `--max-dict`    | 1.000.000   | Tamanho máximo do dicionário LZW         |
| `--train`       | train.csv   | Caminho do CSV de treino                 |
| `--test`        | test.csv    | Caminho do CSV de teste                  |
| `--save-model`  | —           | Salvar modelos em arquivo pickle         |
| `--load-model`  | —           | Carregar modelos de arquivo pickle       |

### Rodar todos os experimentos

```bash
python experiments.py
```

Executa automaticamente:
1. PPM variando kmax (2–6) com 5.000 textos/classe
2. PPM kmax=5 variando tamanho de treino (500–30.000)
3. LZW variando tamanho de treino (500–30.000)

Gera `experiment_results.csv` e `experiment_results.png`.

---

## Resultados

| Modelo              | Treino/classe | Acurácia | Tempo  |
|---------------------|:------------:|:--------:|:------:|
| **PPM kmax=5**      | 30.000       | **91,1%** | 162s  |
| PPM kmax=5          | 10.000       | 89,0%    | 73s    |
| LZW                 | 30.000       | 87,8%    | 10s    |
| PPM kmax=3          | 5.000        | 87,4%    | 43s    |
| LZW                 | 5.000        | 84,2%    | 2s     |

### F1-Score por classe (melhor modelo: PPM kmax=5, 30K)

| Classe   | F1     |
|----------|--------|
| World    | 90,8%  |
| Sports   | 97,0%  |
| Business | 87,6%  |
| Sci/Tech | 89,0%  |

---

## Referências

- Slides ITI 2025.2, Prof. Leonardo Vidal Batista
- Cleary, J. G. & Witten, I. H. (1984). *Data Compression Using Adaptive Coding and Partial String Matching*. IEEE Transactions on Communications.
- Frank, E., Chui, C., & Witten, I. H. (2000). *Text Categorization Using Compression Models*. DCC.
