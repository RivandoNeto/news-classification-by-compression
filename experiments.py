"""
experiments.py — Roda experimentos de classificação variando parâmetros
e gera gráficos dos resultados.

Uso:
  python experiments.py
"""

import csv
import os
import sys
import time

# Adicionar o diretório atual ao path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classify import (
    CLASS_NAMES, PPMCModel, LZWModel,
    load_csv, group_by_class, train_models, evaluate,
)


RESULTS_CSV = "experiment_results.csv"


def run_experiment(method, groups, test_data, kmax=5, rho=1,
                   max_train=None, max_dict=1_000_000):
    """Roda um experimento e retorna dict com resultados."""
    label = f"{method.upper()} kmax={kmax}" if method == "ppm" else "LZW"
    mt = max_train if max_train else len(list(groups.values())[0])
    print(f"\n>>> {label}, max_train={mt:,} ...")

    t0 = time.time()
    models = train_models(
        groups, method, kmax=kmax, rho=rho,
        max_train=max_train, max_dict=max_dict,
    )
    train_time = time.time() - t0

    t0 = time.time()
    accuracy, confusion, _ = evaluate(models, test_data)
    eval_time = time.time() - t0

    # Métricas por classe
    classes = sorted(groups.keys())
    per_class = {}
    for cls in classes:
        tp = confusion.get((cls, cls), 0)
        fp = sum(confusion.get((o, cls), 0) for o in classes if o != cls)
        fn = sum(confusion.get((cls, o), 0) for o in classes if o != cls)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        per_class[cls] = {"precision": prec, "recall": rec, "f1": f1}

    result = {
        "method": method,
        "kmax": kmax if method == "ppm" else "-",
        "max_train": mt,
        "accuracy": accuracy,
        "train_time": train_time,
        "eval_time": eval_time,
    }
    for cls in classes:
        name = CLASS_NAMES.get(cls, str(cls))
        result[f"f1_{name}"] = per_class[cls]["f1"]

    # Liberar memória dos modelos
    del models

    print(f"  => Acuracia: {100*accuracy:.2f}%, "
          f"treino: {train_time:.1f}s, avaliacao: {eval_time:.1f}s")
    return result


def main():
    print("=" * 65)
    print("  Experimentos de Classificacao por Compressao")
    print("=" * 65)

    # Carregar dados uma vez
    print("\nCarregando dados ...")
    train_data = load_csv("train.csv")
    test_data = load_csv("test.csv")
    groups = group_by_class(train_data)
    print(f"  Treino: {len(train_data):,} | Teste: {len(test_data):,}\n")

    results = []

    # Já temos resultados anteriores, mas vamos rodar tudo sistematicamente

    # ── Exp 1: PPM variando kmax (com 5000 treino p/ ser rápido) ─────
    print("\n" + "=" * 65)
    print("  Exp 1: PPM variando kmax (max_train=5000)")
    print("=" * 65)
    for kmax in [2, 3, 4, 5, 6]:
        r = run_experiment("ppm", groups, test_data, kmax=kmax, max_train=5000)
        results.append(r)

    # ── Exp 2: PPM kmax=5 variando tamanho de treino ─────────────────
    print("\n" + "=" * 65)
    print("  Exp 2: PPM kmax=5 variando tamanho de treino")
    print("=" * 65)
    for mt in [500, 1000, 2000, 5000, 10000, 30000]:
        # kmax=5 com 5000 já foi rodado acima, pular
        if mt == 5000:
            continue
        r = run_experiment("ppm", groups, test_data, kmax=5, max_train=mt)
        results.append(r)

    # ── Exp 3: LZW variando tamanho de treino ────────────────────────
    print("\n" + "=" * 65)
    print("  Exp 3: LZW variando tamanho de treino")
    print("=" * 65)
    for mt in [500, 1000, 5000, 10000, 30000]:
        r = run_experiment("lzw", groups, test_data, max_train=mt)
        results.append(r)

    # ── Salvar resultados em CSV ─────────────────────────────────────
    fieldnames = list(results[0].keys())
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResultados salvos em {RESULTS_CSV}")

    # ── Tabela resumo ────────────────────────────────────────────────
    print(f"\n{'=' * 75}")
    print(f"  RESUMO DE TODOS OS EXPERIMENTOS")
    print(f"{'=' * 75}")
    print(f"  {'Metodo':<12} {'kmax':>5} {'Treino':>8} {'Acc%':>8} "
          f"{'F1-Wor':>7} {'F1-Spo':>7} {'F1-Bus':>7} {'F1-ScT':>7} "
          f"{'Tempo':>7}")
    print(f"  {'-' * 72}")
    for r in sorted(results, key=lambda x: -x["accuracy"]):
        print(f"  {r['method'].upper():<12} {str(r['kmax']):>5} "
              f"{r['max_train']:>8} {100*r['accuracy']:>7.2f}% "
              f"{100*r.get('f1_World',0):>6.1f} {100*r.get('f1_Sports',0):>6.1f} "
              f"{100*r.get('f1_Business',0):>6.1f} {100*r.get('f1_Sci/Tech',0):>6.1f} "
              f"{r['train_time']+r['eval_time']:>6.0f}s")

    # ── Gerar gráficos ───────────────────────────────────────────────
    generate_plots(results)


def generate_plots(results):
    """Gera gráficos dos resultados."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[!] matplotlib nao encontrado. Instale com: pip install matplotlib")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Gráfico 1: Acurácia vs kmax (PPM, 5000 treino) ──────────────
    ax = axes[0]
    ppm_kmax = [(r["kmax"], r["accuracy"]) for r in results
                if r["method"] == "ppm" and r["max_train"] == 5000]
    ppm_kmax.sort()
    if ppm_kmax:
        ks, accs = zip(*ppm_kmax)
        ax.plot(ks, [a * 100 for a in accs], "bo-", linewidth=2, markersize=8)
        for k, a in zip(ks, accs):
            ax.annotate(f"{a*100:.1f}%", (k, a*100), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=9)
    ax.set_xlabel("kmax (ordem maxima)", fontsize=11)
    ax.set_ylabel("Acuracia (%)", fontsize=11)
    ax.set_title("PPM-C: Acuracia vs kmax\n(5.000 textos/classe)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(70, 95)

    # ── Gráfico 2: Acurácia vs tamanho de treino (PPM kmax=5) ───────
    ax = axes[1]
    ppm_size = [(r["max_train"], r["accuracy"]) for r in results
                if r["method"] == "ppm" and r["kmax"] == 5]
    ppm_size.sort()

    lzw_size = [(r["max_train"], r["accuracy"]) for r in results
                if r["method"] == "lzw"]
    lzw_size.sort()

    if ppm_size:
        xs, accs = zip(*ppm_size)
        ax.plot(xs, [a * 100 for a in accs], "bo-", linewidth=2, markersize=8,
                label="PPM-C (kmax=5)")
    if lzw_size:
        xs, accs = zip(*lzw_size)
        ax.plot(xs, [a * 100 for a in accs], "rs--", linewidth=2, markersize=8,
                label="LZW")

    ax.set_xlabel("Textos de treino por classe", fontsize=11)
    ax.set_ylabel("Acuracia (%)", fontsize=11)
    ax.set_title("Acuracia vs Tamanho do Treino", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")
    ax.set_ylim(70, 95)

    # ── Gráfico 3: F1 por classe (melhor PPM vs melhor LZW) ─────────
    ax = axes[2]
    best_ppm = max([r for r in results if r["method"] == "ppm"],
                   key=lambda x: x["accuracy"], default=None)
    best_lzw = max([r for r in results if r["method"] == "lzw"],
                   key=lambda x: x["accuracy"], default=None)

    classes = ["World", "Sports", "Business", "Sci/Tech"]
    x_pos = range(len(classes))
    width = 0.35

    if best_ppm:
        f1_ppm = [best_ppm.get(f"f1_{c}", 0) * 100 for c in classes]
        bars1 = ax.bar([p - width/2 for p in x_pos], f1_ppm, width,
                       label=f"PPM kmax={best_ppm['kmax']} ({best_ppm['max_train']})",
                       color="steelblue")
        for bar, v in zip(bars1, f1_ppm):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", fontsize=8)

    if best_lzw:
        f1_lzw = [best_lzw.get(f"f1_{c}", 0) * 100 for c in classes]
        bars2 = ax.bar([p + width/2 for p in x_pos], f1_lzw, width,
                       label=f"LZW ({best_lzw['max_train']})",
                       color="indianred")
        for bar, v in zip(bars2, f1_lzw):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", fontsize=8)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("F1-Score (%)", fontsize=11)
    ax.set_title("F1 por Classe (melhores modelos)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(60, 100)

    plt.tight_layout()
    plt.savefig("experiment_results.png", dpi=150, bbox_inches="tight")
    print(f"\nGrafico salvo em experiment_results.png")


if __name__ == "__main__":
    main()
