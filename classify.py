"""
classify.py — Classificação de notícias usando compressão (PPM-C e LZW)
Projeto 2 — ITI 2025.2

Abordagem:
  1. Treina um modelo de compressão por classe (concatenando todos os textos)
  2. Para cada texto de teste, calcula o comprimento de código sob cada modelo
  3. Classifica pela menor entropia cruzada (menor comprimento de código)

Uso:
  python classify.py ppm --kmax 5 --train train.csv --test test.csv
  python classify.py lzw --train train.csv --test test.csv
  python classify.py ppm --kmax 3 --max-train 1000 --train train.csv --test test.csv
"""

import argparse
import csv
import math
import os
import pickle
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Constantes
# =============================================================================

ESC = 256
CLASS_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


# =============================================================================
# Modelo PPM-C para classificação
# =============================================================================

class PPMCModel:
    """
    Modelo PPM-C para classificação por compressão.

    Treina em dados de uma classe, construindo tabelas de frequência para
    contextos de ordem 0 até kmax. Para classificar, calcula o comprimento
    de código (entropia cruzada) de um texto de teste usando as tabelas
    congeladas (sem atualização durante a classificação).

    Mecanismo de exclusão (PPM-C):
        Quando um símbolo não é encontrado no contexto de ordem k,
        codifica-se o escape e os símbolos deste contexto são excluídos
        nas ordens inferiores.

    Massa de escape (método C):
        P(ESC | ctx) = rho * d / (total + rho * d)
        onde d = número de símbolos distintos no contexto (após exclusão).
    """

    def __init__(self, kmax: int, rho: int = 1):
        self.kmax = kmax
        self.rho = rho
        self.contexts: Dict[bytes, Dict[int, int]] = {}

    def train(self, data: bytes):
        """Alimenta o modelo com dados de treino, byte a byte."""
        kmax = self.kmax
        contexts = self.contexts
        hist = bytearray()

        for byte_val in data:
            h_len = len(hist)
            max_k = min(kmax, h_len)

            for k in range(max_k + 1):
                ctx = bytes(hist[h_len - k:]) if k > 0 else b""
                d = contexts.get(ctx)
                if d is None:
                    d = {}
                    contexts[ctx] = d
                d[byte_val] = d.get(byte_val, 0) + 1

            # Manter apenas os últimos kmax bytes no histórico
            if h_len < kmax:
                hist.append(byte_val)
            else:
                del hist[0]
                hist.append(byte_val)

    def score(self, data: bytes) -> float:
        """
        Calcula o comprimento de código (em bits) do texto sob este modelo.

        Usa as tabelas de frequência congeladas (construídas no treino).
        Mantém um histórico local para acompanhar o contexto do texto de teste.
        """
        kmax = self.kmax
        rho = self.rho
        contexts = self.contexts
        hist = bytearray()
        total_bits = 0.0
        _log2 = math.log2

        for byte_val in data:
            excluded = set()
            h_len = len(hist)
            max_k = min(kmax, h_len)
            symbol_coded = False

            for k in range(max_k, -1, -1):
                ctx = bytes(hist[h_len - k:]) if k > 0 else b""
                counts = contexts.get(ctx)

                if counts is None:
                    # Contexto não existe no modelo → escape grátis
                    continue

                # Contar símbolos disponíveis (não excluídos) e verificar byte_val
                avail_count = 0
                avail_total = 0
                sym_count = 0
                has_symbol = False

                for s, c in counts.items():
                    if s not in excluded:
                        avail_count += 1
                        avail_total += c
                        if s == byte_val:
                            sym_count = c
                            has_symbol = True

                if avail_count == 0:
                    # Todos os símbolos deste contexto já foram excluídos
                    continue

                esc_mass = rho * avail_count
                total = avail_total + esc_mass

                if has_symbol:
                    # Símbolo encontrado nesta ordem
                    total_bits -= _log2(sym_count / total)
                    symbol_coded = True
                    break
                else:
                    # Escape: descer para a próxima ordem
                    total_bits -= _log2(esc_mass / total)
                    excluded.update(s for s in counts if s not in excluded)

            if not symbol_coded:
                # Ordem -1: distribuição uniforme sobre símbolos restantes
                remaining = 256 - len(excluded)
                if remaining > 0:
                    total_bits += _log2(remaining)

            # Atualizar histórico local
            if h_len < kmax:
                hist.append(byte_val)
            else:
                del hist[0]
                hist.append(byte_val)

        return total_bits


# =============================================================================
# Modelo LZW para classificação
# =============================================================================

class LZWModel:
    """
    Modelo LZW para classificação por compressão.

    Constrói uma trie (dicionário) a partir dos dados de treino de uma classe.
    Para classificar, usa o dicionário congelado para codificar o texto de teste
    e conta o número de códigos de saída. Textos que casam melhor com os padrões
    da classe produzem menos códigos (melhor compressão).
    """

    def __init__(self, max_entries: int = 1_000_000):
        # Trie: cada nó é um dict mapeando byte -> nó filho
        # Raiz tem 256 filhos (um por byte)
        self.root: Dict[int, dict] = {i: {} for i in range(256)}
        self.n_entries = 256
        self.max_entries = max_entries

    def train(self, data: bytes):
        """Constrói o dicionário LZW a partir dos dados de treino."""
        root = self.root
        node = root
        max_entries = self.max_entries
        n_entries = self.n_entries

        for byte_val in data:
            if byte_val in node:
                node = node[byte_val]
            else:
                if n_entries < max_entries:
                    node[byte_val] = {}
                    n_entries += 1
                node = root[byte_val]

        self.n_entries = n_entries

    def score(self, data: bytes) -> float:
        """
        Conta o número de códigos de saída usando o dicionário congelado.
        Retorna o comprimento estimado em bits (n_códigos × log2(tam_dicionário)).
        """
        if not data:
            return 0.0

        root = self.root
        node = root
        n_codes = 0

        for byte_val in data:
            if byte_val in node:
                node = node[byte_val]
            else:
                n_codes += 1
                node = root[byte_val]

        n_codes += 1  # último código pendente

        # Comprimento em bits: cada código precisa de log2(dict_size) bits
        if self.n_entries > 1:
            return n_codes * math.log2(self.n_entries)
        return float(n_codes)


# =============================================================================
# Carregamento de dados
# =============================================================================

def load_csv(path: str) -> List[Tuple[int, str]]:
    """Carrega CSV do AG News: retorna lista de (classe, texto)."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # pular cabeçalho
        for row in reader:
            cls = int(row[0])
            title = row[1]
            desc = row[2] if len(row) > 2 else ""
            text = f"{title} {desc}".strip()
            data.append((cls, text))
    return data


def group_by_class(data: List[Tuple[int, str]]) -> Dict[int, List[str]]:
    """Agrupa textos por classe."""
    groups: Dict[int, List[str]] = {}
    for cls, text in data:
        groups.setdefault(cls, []).append(text)
    return groups


# =============================================================================
# Treinamento
# =============================================================================

def train_models(
    groups: Dict[int, List[str]],
    method: str,
    kmax: int = 5,
    rho: int = 1,
    max_train: Optional[int] = None,
    max_dict: int = 1_000_000,
) -> Dict[int, object]:
    """Treina um modelo por classe."""
    models = {}

    for cls in sorted(groups):
        texts = groups[cls]
        if max_train is not None and max_train < len(texts):
            texts = texts[:max_train]

        # Concatenar textos com separador newline
        corpus = "\n".join(texts).encode("utf-8")

        name = CLASS_NAMES.get(cls, "?")
        print(f"  Classe {cls} ({name}): "
              f"{len(texts):,} textos, {len(corpus):,} bytes ... ",
              end="", flush=True)

        t0 = time.time()
        if method == "ppm":
            model = PPMCModel(kmax=kmax, rho=rho)
        else:
            model = LZWModel(max_entries=max_dict)

        model.train(corpus)
        dt = time.time() - t0

        if method == "ppm":
            print(f"OK ({len(model.contexts):,} contextos, {dt:.1f}s)")
        else:
            print(f"OK ({model.n_entries:,} entradas, {dt:.1f}s)")

        models[cls] = model

    return models


# =============================================================================
# Classificação e avaliação
# =============================================================================

def classify_text(
    models: Dict[int, object],
    text: str,
) -> Tuple[int, Dict[int, float]]:
    """Classifica um texto, retornando (classe_predita, scores_por_classe)."""
    data = text.encode("utf-8")
    scores = {}
    for cls, model in models.items():
        scores[cls] = model.score(data)

    predicted = min(scores, key=scores.get)
    return predicted, scores


def evaluate(
    models: Dict[int, object],
    test_data: List[Tuple[int, str]],
) -> Tuple[float, Dict[Tuple[int, int], int], List[Tuple[int, str, int, Dict[int, float]]]]:
    """
    Avalia nos dados de teste.
    Retorna: (acurácia, matriz_de_confusão, exemplos_errados)
    """
    correct = 0
    total = 0
    confusion: Dict[Tuple[int, int], int] = Counter()
    wrong_examples: List[Tuple[int, str, int, Dict[int, float]]] = []

    n = len(test_data)
    t0 = time.time()

    for i, (true_cls, text) in enumerate(test_data):
        pred_cls, scores = classify_text(models, text)
        confusion[(true_cls, pred_cls)] += 1

        if pred_cls == true_cls:
            correct += 1
        elif len(wrong_examples) < 10:
            wrong_examples.append((true_cls, text, pred_cls, scores))

        total += 1

        if (i + 1) % 200 == 0 or i == n - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (n - i - 1) / rate if rate > 0 else 0
            print(f"\r  Avaliando: {i+1}/{n} "
                  f"({100 * correct / total:.1f}% acc, "
                  f"{rate:.0f} textos/s, "
                  f"ETA {eta:.0f}s)   ", end="", flush=True)

    print()
    accuracy = correct / total if total > 0 else 0
    return accuracy, confusion, wrong_examples


# =============================================================================
# Exibição de resultados
# =============================================================================

def print_results(
    accuracy: float,
    confusion: Dict[Tuple[int, int], int],
    classes: List[int],
    wrong_examples: List[Tuple[int, str, int, Dict[int, float]]],
    method: str,
    params: str,
):
    """Imprime resultados detalhados."""
    print(f"\n{'=' * 65}")
    print(f"  RESULTADOS — {method.upper()} {params}")
    print(f"{'=' * 65}")
    print(f"\n  Acurácia geral: {100 * accuracy:.2f}%\n")

    # Métricas por classe
    print(f"  {'Classe':<15} {'Precisão':>10} {'Recall':>10} "
          f"{'F1':>10} {'Suporte':>9}")
    print(f"  {'-' * 55}")

    for cls in classes:
        tp = confusion.get((cls, cls), 0)
        fp = sum(confusion.get((o, cls), 0) for o in classes if o != cls)
        fn = sum(confusion.get((cls, o), 0) for o in classes if o != cls)
        support = tp + fn

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        name = CLASS_NAMES.get(cls, str(cls))
        print(f"  {cls} {name:<13} {100 * prec:>9.1f}% {100 * rec:>9.1f}% "
              f"{100 * f1:>9.1f}% {support:>9}")

    # Matriz de confusão
    print(f"\n  Matriz de Confusão (linhas = real, colunas = predito):\n")
    print(f"  {'':>15}", end="")
    for cls in classes:
        name = CLASS_NAMES.get(cls, str(cls))
        print(f" {name:>10}", end="")
    print()

    for true_cls in classes:
        name = CLASS_NAMES.get(true_cls, str(true_cls))
        print(f"  {true_cls} {name:<13}", end="")
        for pred_cls in classes:
            count = confusion.get((true_cls, pred_cls), 0)
            print(f" {count:>10}", end="")
        print()

    # Exemplos de erros
    if wrong_examples:
        print(f"\n  Exemplos de erros de classificação:")
        print(f"  {'-' * 55}")
        for true_cls, text, pred_cls, scores in wrong_examples[:5]:
            true_name = CLASS_NAMES.get(true_cls, str(true_cls))
            pred_name = CLASS_NAMES.get(pred_cls, str(pred_cls))
            short = text[:80] + ("..." if len(text) > 80 else "")
            print(f"  Real: {true_name} -> Predito: {pred_name}")
            print(f"    \"{short}\"")

            # Bits por byte para cada classe
            n_bytes = len(text.encode("utf-8"))
            bpb = {c: s / n_bytes for c, s in scores.items()}
            score_str = "  ".join(
                f"{CLASS_NAMES.get(c, str(c))}={bpb[c]:.2f}"
                for c in sorted(scores)
            )
            print(f"    bpb: {score_str}")
            print()


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Classificação de notícias por compressão (PPM-C / LZW) — ITI 2025.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python classify.py ppm --kmax 5
  python classify.py ppm --kmax 3 --max-train 1000
  python classify.py lzw --max-dict 500000
  python classify.py ppm --kmax 5 --save-model modelos_ppm5.pkl
  python classify.py ppm --load-model modelos_ppm5.pkl
        """,
    )
    parser.add_argument(
        "method", choices=["ppm", "lzw"],
        help="Método de compressão (ppm ou lzw)",
    )
    parser.add_argument(
        "--train", default="train.csv",
        help="Caminho do CSV de treino (default: train.csv)",
    )
    parser.add_argument(
        "--test", default="test.csv",
        help="Caminho do CSV de teste (default: test.csv)",
    )
    parser.add_argument(
        "--kmax", type=int, default=5,
        help="Ordem máxima do PPM (default: 5)",
    )
    parser.add_argument(
        "--rho", type=int, default=1,
        help="Parâmetro rho do PPM-C (default: 1)",
    )
    parser.add_argument(
        "--max-train", type=int, default=None,
        help="Máximo de textos de treino por classe (default: todos)",
    )
    parser.add_argument(
        "--max-dict", type=int, default=1_000_000,
        help="Tamanho máximo do dicionário LZW (default: 1.000.000)",
    )
    parser.add_argument(
        "--save-model", type=str, default=None,
        help="Salvar modelos treinados em arquivo pickle",
    )
    parser.add_argument(
        "--load-model", type=str, default=None,
        help="Carregar modelos de arquivo pickle (pula treino)",
    )

    args = parser.parse_args()

    print(f"\n{'=' * 65}")
    print(f"  Classificação de Notícias por Compressão — ITI 2025.2")
    print(f"{'=' * 65}\n")

    # ── Carregar ou treinar modelos ──────────────────────────────────────

    if args.load_model and os.path.exists(args.load_model):
        print(f"Carregando modelos de {args.load_model} ...")
        with open(args.load_model, "rb") as f:
            models = pickle.load(f)
        print(f"  {len(models)} modelos carregados.\n")
    else:
        print(f"Carregando dados de treino ({args.train}) ...")
        train_data = load_csv(args.train)
        groups = group_by_class(train_data)

        for cls in sorted(groups):
            name = CLASS_NAMES.get(cls, "?")
            print(f"  Classe {cls} ({name}): {len(groups[cls]):,} textos")
        print()

        params_str = ""
        if args.method == "ppm":
            params_str = f"kmax={args.kmax}, rho={args.rho}"
        else:
            params_str = f"max_dict={args.max_dict:,}"

        if args.max_train:
            params_str += f", max_train={args.max_train:,}"

        print(f"Treinando modelos {args.method.upper()} ({params_str}) ...\n")

        t_total = time.time()
        models = train_models(
            groups,
            method=args.method,
            kmax=args.kmax,
            rho=args.rho,
            max_train=args.max_train,
            max_dict=args.max_dict,
        )
        dt_total = time.time() - t_total
        print(f"\n  Treino total: {dt_total:.1f}s\n")

        if args.save_model:
            print(f"Salvando modelos em {args.save_model} ...")
            with open(args.save_model, "wb") as f:
                pickle.dump(models, f)
            size_mb = os.path.getsize(args.save_model) / (1024 * 1024)
            print(f"  OK ({size_mb:.1f} MB)\n")

    # ── Avaliar ──────────────────────────────────────────────────────────

    print(f"Carregando dados de teste ({args.test}) ...")
    test_data = load_csv(args.test)
    print(f"  {len(test_data):,} textos de teste.\n")

    print("Classificando ...\n")
    t_eval = time.time()
    accuracy, confusion, wrong_examples = evaluate(models, test_data)
    dt_eval = time.time() - t_eval

    classes = sorted(set(cls for cls, _ in test_data))

    params_str = ""
    if args.method == "ppm":
        params_str = f"kmax={args.kmax}, rho={args.rho}"
    else:
        params_str = f"max_dict={args.max_dict:,}"
    if args.max_train:
        params_str += f", max_train={args.max_train:,}"

    print_results(accuracy, confusion, classes, wrong_examples,
                  args.method, params_str)

    print(f"\n  Tempo de avaliação: {dt_eval:.1f}s")
    print(f"  Velocidade: {len(test_data) / dt_eval:.0f} textos/s\n")


if __name__ == "__main__":
    main()
