"""
main.py — Compressor/Descompressor PPM-C para ITI 2025.2

Algoritmo: PPM-C (Prediction by Partial Matching com mecanismo de exclusão)
Codificação: Range Coding aritmético de 64 bits

Subcomandos:
  c       — Comprimir um arquivo (opcionalmente gera gráfico progressivo)
  d       — Descomprimir um arquivo
  bench   — Rodar Kmax 0..N e gerar tabela de resultados (req. 3.1)
  concat  — Concatenar arquivos do Silesia em stream único
  verify  — Verificar integridade por comparação binária

Exemplos:
  python main.py c dickens dickens_k6 --kmax 6
  python main.py c dickens dickens_k6 --kmax 6 --graph
  python main.py c silesia.tar silesia_k6 --kmax 6 --graph --manifest silesia.bin.manifest.json
  python main.py d dickens_k6 dickens_out --verify dickens
  python main.py bench dickens --kmax-max 10 --csv resultados.csv
  python main.py concat silesia.bin --files ../silesia/*
  python main.py verify dickens dickens_out

Log de execucoes: ppmc_execucoes.log (mesmo diretorio do script, modo append)
"""

import argparse
import json
import logging
import os
import struct
import sys
import time
from bisect import bisect_right
from typing import List, Optional, Tuple


# =============================================================================
# Logger (append ao arquivo de log — nunca sobrescreve)
# =============================================================================

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppmc_execucoes.log")

def _setup_logger() -> logging.Logger:
    logger = logging.getLogger("ppmc_teste")
    if logger.handlers:
        return logger
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(fh)
    return logger


# =============================================================================
# Bit I/O eficiente (direto em bytearray, sem expansão em lista de inteiros)
# =============================================================================

class BitWriter:
    """Empacota bits diretamente em bytes (MSB first)."""

    def __init__(self):
        self._buf   = bytearray()
        self._cur   = 0
        self._nbits = 0
        self._total = 0

    def write_bit(self, b: int):
        self._cur    = (self._cur << 1) | (1 if b else 0)
        self._nbits += 1
        self._total += 1
        if self._nbits == 8:
            self._buf.append(self._cur)
            self._cur   = 0
            self._nbits = 0

    @property
    def total_bits(self) -> int:
        return self._total

    def get_bytes(self) -> bytes:
        buf = bytearray(self._buf)
        if self._nbits > 0:
            buf.append(self._cur << (8 - self._nbits))
        return bytes(buf)


class BitReader:
    """Lê bits diretamente de um buffer de bytes (MSB first)."""

    def __init__(self, data: bytes, n_bits: int):
        self._data   = data
        self._n_bits = n_bits
        self._pos    = 0

    def read_bit(self) -> int:
        if self._pos >= self._n_bits:
            return 0
        byte_idx = self._pos >> 3
        if byte_idx >= len(self._data):
            return 0
        bit       = (self._data[byte_idx] >> (7 - (self._pos & 7))) & 1
        self._pos += 1
        return bit


# =============================================================================
# Range Coder de 64 bits (codificação aritmética com escalonamento por bits)
# =============================================================================

_RC_TOP  = (1 << 64) - 1
_RC_HALF = 1 << 63
_RC_QTR  = 1 << 62
_RC_TQT  = _RC_QTR * 3


class RangeEncoder:

    def __init__(self, bw: BitWriter):
        self.bw      = bw
        self.low     = 0
        self.high    = _RC_TOP
        self.pending = 0

    def _emit(self, bit: int):
        self.bw.write_bit(bit)
        inv = 1 - bit
        for _ in range(self.pending):
            self.bw.write_bit(inv)
        self.pending = 0

    def update(self, cum_low: int, cum_high: int, cum_total: int):
        rng       = self.high - self.low + 1
        self.high = self.low + (rng * cum_high // cum_total) - 1
        self.low  = self.low + (rng * cum_low  // cum_total)

        while True:
            if self.high < _RC_HALF:
                self._emit(0)
            elif self.low >= _RC_HALF:
                self._emit(1)
                self.low  -= _RC_HALF
                self.high -= _RC_HALF
            elif self.low >= _RC_QTR and self.high < _RC_TQT:
                self.pending += 1
                self.low     -= _RC_QTR
                self.high    -= _RC_QTR
            else:
                break
            self.low  = (self.low  << 1) & _RC_TOP
            self.high = ((self.high << 1) | 1) & _RC_TOP

    def finish(self):
        self.pending += 1
        if self.low < _RC_QTR:
            self._emit(0)
        else:
            self._emit(1)


class RangeDecoder:

    def __init__(self, br: BitReader):
        self.br   = br
        self.low  = 0
        self.high = _RC_TOP
        self.code = 0
        for _ in range(64):
            self.code = ((self.code << 1) | self.br.read_bit()) & _RC_TOP

    def _renorm(self):
        while True:
            if self.high < _RC_HALF:
                pass
            elif self.low >= _RC_HALF:
                self.low  -= _RC_HALF
                self.high -= _RC_HALF
                self.code -= _RC_HALF
            elif self.low >= _RC_QTR and self.high < _RC_TQT:
                self.low  -= _RC_QTR
                self.high -= _RC_QTR
                self.code -= _RC_QTR
            else:
                break
            self.low  = (self.low  << 1) & _RC_TOP
            self.high = ((self.high << 1) | 1) & _RC_TOP
            self.code = ((self.code << 1) | self.br.read_bit()) & _RC_TOP

    def get_target(self, cum_total: int) -> int:
        rng = self.high - self.low + 1
        return ((self.code - self.low + 1) * cum_total - 1) // rng

    def update(self, cum_low: int, cum_high: int, cum_total: int):
        rng       = self.high - self.low + 1
        self.high = self.low + (rng * cum_high // cum_total) - 1
        self.low  = self.low + (rng * cum_low  // cum_total)
        self._renorm()


# =============================================================================
# Modelo PPM-C (Prediction by Partial Matching com Exclusão)
# =============================================================================

ESC = 256


class PPMModel:
    """
    Mantém tabelas de frequência para contextos de ordem 0 até kmax.

    Mecanismo de exclusão (PPM-C):
        Quando um símbolo não é encontrado no contexto de ordem k,
        codifica-se o escape e os símbolos deste contexto são adicionados
        ao conjunto de excluídos.

    Massa de escape (método C):
        P(ESC | ctx) = rho * d / (total + rho * d)
        onde d = número de símbolos distintos no contexto (após exclusão).
    """

    def __init__(self, kmax: int, rho: int):
        self.kmax     = kmax
        self.rho      = rho
        self.contexts: dict = {}
        self.hist     = bytearray()

    def reset(self):
        self.contexts.clear()
        self.hist = bytearray()

    def _ctx(self, k: int) -> bytes:
        if k == 0:
            return b""
        n = len(self.hist)
        return bytes(self.hist[-k:]) if n >= k else bytes(self.hist)

    def get_counts(self, ctx: bytes) -> dict:
        return self.contexts.get(ctx, {})

    def update_model(self, symbol: int):
        max_k = min(self.kmax, len(self.hist))
        for k in range(max_k + 1):
            ctx = self._ctx(k)
            d   = self.contexts.setdefault(ctx, {})
            d[symbol] = d.get(symbol, 0) + 1
        self.hist.append(symbol)

    def build_cdf(self, counts: dict, excluded: set):
        avail    = sorted(s for s in counts if s not in excluded)
        d        = len(avail)
        esc_mass = self.rho * d if d > 0 else self.rho

        symbols = avail + [ESC]
        cum     = [0]
        total   = 0
        for s in avail:
            total += counts[s]
            cum.append(total)
        total += esc_mass
        cum.append(total)

        return symbols, cum, total


def _cdf_lookup(symbols: list, cum: list, t: int):
    i = bisect_right(cum, t) - 1
    return symbols[i], cum[i], cum[i + 1]


# =============================================================================
# Codificação PPM-C
# =============================================================================

def ppmc_encode(
    data: bytes,
    kmax: int,
    rho: int,
    j: int = 1000,
    pct: float = 0.30,
    sample_stride: int = 0,
) -> Tuple[bytes, int, List[int], List[Tuple[int, float]]]:
    """
    Comprime `data` com PPM-C + range coding.

    Parâmetros:
        j             : tamanho da janela de monitoramento (0 = reset desativado)
        pct           : limiar de degradação p/ reset (ex: 0.10 = 10%)
        sample_stride : amostrar comprimento progressivo a cada N símbolos (0 = não amostrar)

    Retorna:
        payload         : bytes comprimidos
        n_bits          : bits válidos no payload
        reset_positions : índices de símbolo onde ocorreu reset
        progress        : lista de (n, bps) — comprimento médio progressivo amostrado
    """
    model           = PPMModel(kmax=kmax, rho=rho)
    bw              = BitWriter()
    enc             = RangeEncoder(bw)
    n               = len(data)
    reset_positions = []
    progress        = []

    prev_win_avg   = None
    win_bits_start = 0
    win_sym_count  = 0

    for i, byte_val in enumerate(data):

        # Detecção de não-estacionariedade ao completar cada janela
        if j > 0 and win_sym_count == j:
            cur_bits    = bw.total_bits
            cur_win_avg = (cur_bits - win_bits_start) / j

            if prev_win_avg is not None and cur_win_avg > prev_win_avg * (1.0 + pct):
                reset_positions.append(i)
                model.reset()
                prev_win_avg = None
            else:
                prev_win_avg = cur_win_avg

            win_bits_start = bw.total_bits
            win_sym_count  = 0

        # Codificação do símbolo atual (PPM-C com exclusão)
        s        = byte_val
        excluded = set()
        emitted  = False
        max_k    = min(kmax, len(model.hist))

        for k in range(max_k, -1, -1):
            ctx              = model._ctx(k)
            counts           = model.get_counts(ctx)
            symbols, cum, total = model.build_cdf(counts, excluded)

            if s in symbols[:-1]:
                idx = symbols.index(s)
                enc.update(cum[idx], cum[idx + 1], total)
                emitted = True
                break
            else:
                enc.update(cum[-2], cum[-1], total)
                excluded.update(symbols[:-1])

        if not emitted:
            remaining = sorted(x for x in range(256) if x not in excluded)
            idx       = remaining.index(s)
            enc.update(idx, idx + 1, len(remaining))

        model.update_model(s)
        win_sym_count += 1

        # Amostragem do comprimento médio progressivo
        if sample_stride > 0 and (i + 1) % sample_stride == 0:
            progress.append((i + 1, bw.total_bits / (i + 1)))

    # Garante que o ponto final seja amostrado
    if sample_stride > 0 and n > 0:
        if not progress or progress[-1][0] != n:
            progress.append((n, bw.total_bits / n))

    enc.finish()
    return bw.get_bytes(), bw.total_bits, reset_positions, progress


# =============================================================================
# Decodificação PPM-C
# =============================================================================

def ppmc_decode(
    payload: bytes,
    n_bytes: int,
    n_bits: int,
    kmax: int,
    rho: int,
    reset_positions: List[int],
) -> bytes:
    """
    Descomprime `payload`.

    As posições de reset são armazenadas fora de banda (no cabeçalho do arquivo),
    garantindo sincronismo perfeito com o compressor sem custo adicional de bits.
    """
    model     = PPMModel(kmax=kmax, rho=rho)
    br        = BitReader(payload, n_bits)
    dec       = RangeDecoder(br)
    out       = bytearray()
    reset_set = set(reset_positions)

    for i in range(n_bytes):

        if i in reset_set:
            model.reset()

        excluded = set()
        max_k    = min(kmax, len(model.hist))
        s        = None

        for k in range(max_k, -1, -1):
            ctx              = model._ctx(k)
            counts           = model.get_counts(ctx)
            symbols, cum, total = model.build_cdf(counts, excluded)

            t            = dec.get_target(total)
            sym, lo, hi  = _cdf_lookup(symbols, cum, t)
            dec.update(lo, hi, total)

            if sym != ESC:
                s = sym
                break
            excluded.update(symbols[:-1])

        if s is None:
            remaining = sorted(x for x in range(256) if x not in excluded)
            t         = dec.get_target(len(remaining))
            t         = min(t, len(remaining) - 1)
            s         = remaining[t]
            dec.update(t, t + 1, len(remaining))

        out.append(s)
        model.update_model(s)

    return bytes(out)


# =============================================================================
# Formato do arquivo .ppmc
# =============================================================================
#
# Cabeçalho (big-endian):
#   MAGIC    4s   — b"PPMC"
#   kmax     B    — uint8
#   rho      I    — uint32
#   n_bytes  Q    — uint64  (tamanho original em bytes)
#   n_bits   Q    — uint64  (bits válidos no payload)
#   j        I    — uint32  (tamanho da janela; 0 = reset desativado)
#   pct      f    — float32 (limiar de degradação)
#   n_resets I    — uint32  (número de posições de reset que se seguem)
#
# Seguido de:
#   n_resets × uint64  — posições de reset (índices no dado original)
#
# Seguido do payload comprimido.

MAGIC         = b"PPMC"
HEADER_STRUCT = struct.Struct(">4sB I Q Q I f I")


def compress_file(
    in_path: str,
    out_path: str,
    kmax: int,
    rho: int = 1,
    j: int = 1000,
    pct: float = 0.30,
    sample_stride: int = 0,
    verbose: bool = True,
) -> Tuple[float, int, int, List[int], List[Tuple[int, float]]]:
    """
    Comprime in_path → out_path.
    Retorna (dt, n_bytes_orig, n_bits, reset_positions, progress).
    """
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Arquivo não encontrado: '{in_path}'")
    data = open(in_path, "rb").read()
    if not data:
        raise ValueError("Arquivo de entrada vazio.")

    if verbose:
        sys.stderr.write(
            f"Comprimindo '{os.path.basename(in_path)}' "
            f"({len(data):,} bytes, kmax={kmax}, rho={rho}, j={j}, pct={pct:.0%})...\n"
        )
        sys.stderr.flush()

    t0 = time.perf_counter()
    payload, n_bits, resets, progress = ppmc_encode(
        data, kmax=kmax, rho=rho, j=j, pct=pct, sample_stride=sample_stride
    )
    t1 = time.perf_counter()

    n_resets   = len(resets)
    header     = HEADER_STRUCT.pack(
        MAGIC, kmax & 0xFF, rho & 0xFFFFFFFF,
        len(data), n_bits,
        j & 0xFFFFFFFF, float(pct), n_resets,
    )
    reset_data = struct.pack(f">{n_resets}Q", *resets) if n_resets else b""

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(reset_data)
        f.write(payload)

    return (t1 - t0), len(data), n_bits, resets, progress


def decompress_file(
    in_path: str,
    out_path: str,
    verbose: bool = True,
) -> Tuple[float, int]:
    """
    Descomprime in_path → out_path.
    Retorna (dt, n_bytes_orig).
    """
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Arquivo não encontrado: '{in_path}'")
    raw = open(in_path, "rb").read()
    hdr = HEADER_STRUCT.size

    if len(raw) < hdr:
        raise ValueError("Arquivo muito pequeno para ser PPMC válido.")

    magic, kmax, rho, n_bytes, n_bits, j, pct, n_resets = HEADER_STRUCT.unpack(raw[:hdr])

    if magic != MAGIC:
        raise ValueError(f"Cabeçalho inválido (magic={magic!r}).")
    if n_bytes == 0:
        raise ValueError("O arquivo indica 0 bytes originais.")

    if verbose:
        sys.stderr.write(
            f"Descomprimindo '{os.path.basename(in_path)}' "
            f"({n_bytes:,} bytes originais, {n_resets} reset(s))...\n"
        )
        sys.stderr.flush()

    off = hdr
    if n_resets > 0:
        fmt    = f">{n_resets}Q"
        sz     = struct.calcsize(fmt)
        if off + sz > len(raw):
            raise ValueError("Arquivo corrompido: dados de reset ausentes.")
        resets = list(struct.unpack(fmt, raw[off:off + sz]))
        off   += sz
    else:
        resets = []

    t0   = time.perf_counter()
    data = ppmc_decode(raw[off:], n_bytes, n_bits, kmax, rho, resets)
    t1   = time.perf_counter()

    with open(out_path, "wb") as f:
        f.write(data)

    return (t1 - t0), n_bytes


def verify_integrity(original_path: str, restored_path: str) -> bool:
    orig = open(original_path, "rb").read()
    rest = open(restored_path, "rb").read()
    return orig == rest


# =============================================================================
# Geração de gráfico progressivo (integrado à compressão)
# =============================================================================

def generate_graph(
    progress: List[Tuple[int, float]],
    resets: List[int],
    bps_final: float,
    in_basename: str,
    kmax: int,
    rho: int,
    j: int,
    pct: float,
    out_png: str,
    manifest_path: Optional[str] = None,
):
    """Gera o gráfico de comprimento médio progressivo a partir dos dados coletados."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        from matplotlib.lines import Line2D
    except ImportError:
        sys.exit("Erro: matplotlib nao instalado. Execute: pip install matplotlib")

    # Carrega manifesto do Silesia (limites de arquivo para marcação no gráfico)
    file_boundaries: List[Tuple[int, str]] = []
    if manifest_path:
        if not os.path.isfile(manifest_path):
            sys.exit(f"Manifesto nao encontrado: '{manifest_path}'")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        offset = 0
        for entry in manifest:
            file_boundaries.append((offset, entry["name"]))
            offset += entry["size"]

    xs = [p[0] for p in progress]
    ys = [p[1] for p in progress]

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(xs, ys, linewidth=0.9, color="steelblue", label="bps progressivo")

    # Posições de reset (vermelho tracejado)
    for r_pos in resets:
        ax.axvline(r_pos, color="tomato", linewidth=0.8, linestyle="--", alpha=0.75)

    # Limites de arquivo do Silesia (verde pontilhado + rótulo)
    for bnd_offset, fname in file_boundaries:
        if bnd_offset > 0:
            ax.axvline(bnd_offset, color="seagreen", linewidth=1.0, linestyle=":", alpha=0.8)
            ax.text(
                bnd_offset, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else max(ys),
                f" {fname}", fontsize=6, color="seagreen",
                rotation=90, va="top", ha="right",
            )

    # Linha de referência: bps final
    ax.axhline(bps_final, color="gray", linewidth=0.7, linestyle="-.",
               label=f"bps final ({bps_final:.4f})")

    ax.set_xlabel("Posição n (símbolo)")
    ax.set_ylabel("bits acumulados / n  [bps]")
    ax.set_title(
        f"Comprimento Médio Progressivo — {in_basename}\n"
        f"kmax={kmax}, rho={rho}, j={j}, pct={pct:.0%}  "
        f"| bps final={bps_final:.4f} | resets={len(resets)}"
    )

    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x / 1e6:.1f}M" if x >= 1e6 else f"{x / 1e3:.0f}k")
    )
    ax.grid(True, alpha=0.3)

    handles = [Line2D([0], [0], color="steelblue", lw=1.5, label="bps progressivo"),
               Line2D([0], [0], color="gray", lw=0.8, ls="-.", label=f"bps final ({bps_final:.4f})")]
    if resets:
        handles.append(Line2D([0], [0], color="tomato", lw=1, ls="--",
                               label=f"Reset ({len(resets)} total)"))
    if file_boundaries:
        handles.append(Line2D([0], [0], color="seagreen", lw=1, ls=":",
                               label="Limite de arquivo (Silesia)"))
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# =============================================================================
# Utilitários de formatação
# =============================================================================

def _fmt_size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _print_table(rows: list, headers: list):
    col_w = [len(h) for h in headers]
    for row in rows:
        for j_idx, cell in enumerate(row):
            col_w[j_idx] = max(col_w[j_idx], len(str(cell)))
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    fmt = "|" + "|".join(f" {{:<{w}}} " for w in col_w) + "|"
    print(sep)
    print(fmt.format(*headers))
    print(sep)
    for row in rows:
        print(fmt.format(*[str(c) for c in row]))
    print(sep)


# =============================================================================
# Subcomando: bench — Análise de Ordem e Performance (req. 3.1)
# =============================================================================

def cmd_bench(args, log: logging.Logger):
    import tempfile

    in_path  = args.in_path
    kmax_max = args.kmax_max
    rho      = args.rho
    j        = 0 if args.no_reset else args.j
    pct      = args.pct

    log.info(
        f"INICIO bench | in='{in_path}' kmax_max={kmax_max} "
        f"rho={rho} j={j} pct={pct:.0%}"
    )

    print(f"\nBenchmark: '{os.path.basename(in_path)}'")
    print(f"Parametros: rho={rho}, j={j}, pct={pct:.0%}, kmax 0..{kmax_max}\n")

    headers = ["kmax", "bps", "ratio", "tam_comp", "resets", "t_comp(s)", "t_decomp(s)", "OK"]
    rows    = []

    for kmax in range(kmax_max + 1):
        with tempfile.NamedTemporaryFile(suffix=".ppmc", delete=False) as tf:
            comp_path = tf.name
        with tempfile.NamedTemporaryFile(suffix=".dec", delete=False) as tf:
            dec_path = tf.name

        try:
            dt_c, n_orig, n_bits, resets, _ = compress_file(
                in_path, comp_path,
                kmax=kmax, rho=rho, j=j, pct=pct,
                verbose=False,
            )
            bps   = n_bits / n_orig if n_orig else 0.0
            ratio = (n_orig * 8) / n_bits if n_bits else 0.0
            comp_sz = os.path.getsize(comp_path)

            dt_d, _ = decompress_file(comp_path, dec_path, verbose=False)
            ok      = verify_integrity(in_path, dec_path)

            rows.append([
                kmax,
                f"{bps:.4f}",
                f"{ratio:.3f}:1",
                _fmt_size(comp_sz),
                len(resets),
                f"{dt_c:.2f}",
                f"{dt_d:.2f}",
                "OK" if ok else "FALHOU",
            ])
            log.info(
                f"bench kmax={kmax} | bps={bps:.4f} ratio={ratio:.3f}:1 "
                f"resets={len(resets)} t_comp={dt_c:.2f}s t_decomp={dt_d:.2f}s "
                f"integridade={'OK' if ok else 'FALHOU'}"
            )
            print(
                f"  kmax={kmax:2d}  bps={bps:.4f}  ratio={ratio:.3f}:1"
                f"  t_c={dt_c:.2f}s  t_d={dt_d:.2f}s  resets={len(resets)}"
                f"  {'OK' if ok else 'FALHOU'}"
            )

        except Exception as e:
            log.error(f"bench kmax={kmax} | ERRO: {e}")
            raise

        finally:
            for p in (comp_path, dec_path):
                try:
                    os.unlink(p)
                except OSError:
                    pass

    print()
    _print_table(rows, headers)

    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as f:
            f.write(",".join(headers) + "\n")
            for row in rows:
                f.write(",".join(str(c) for c in row) + "\n")
        print(f"\nTabela salva em '{args.csv}'")
        log.info(f"bench | tabela CSV salva em '{args.csv}'")

    log.info(f"FIM bench | in='{in_path}' kmax_max={kmax_max} | {len(rows)} iteracoes concluidas")


# =============================================================================
# Subcomando: concat — Concatenar arquivos do Silesia (req. 3.3)
# =============================================================================

def cmd_concat(args, log: logging.Logger):
    out_path      = args.out_path
    manifest_path = out_path + ".manifest.json"
    manifest      = []
    total         = 0

    log.info(f"INICIO concat | out='{out_path}' n_files={len(args.files)}")

    with open(out_path, "wb") as fout:
        for fpath in sorted(args.files):
            if not os.path.isfile(fpath):
                sys.exit(f"Erro: arquivo nao encontrado: '{fpath}'")
            data = open(fpath, "rb").read()
            fout.write(data)
            sz   = len(data)
            total += sz
            manifest.append({"name": os.path.basename(fpath), "size": sz})
            print(f"  {os.path.basename(fpath):25s}  {_fmt_size(sz)}")

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nTotal concatenado : {_fmt_size(total)}")
    print(f"Arquivo de saida  : '{out_path}'")
    print(f"Manifesto         : '{manifest_path}'")

    log.info(
        f"FIM concat | out='{out_path}' total={_fmt_size(total)} "
        f"n_files={len(manifest)} manifest='{manifest_path}'"
    )


# =============================================================================
# Subcomando: verify — Verificação de integridade (req. 4)
# =============================================================================

def cmd_verify(args, log: logging.Logger):
    if not os.path.isfile(args.original):
        sys.exit(f"Erro: '{args.original}' nao encontrado.")
    if not os.path.isfile(args.restored):
        sys.exit(f"Erro: '{args.restored}' nao encontrado.")

    log.info(f"INICIO verify | original='{args.original}' restored='{args.restored}'")

    if verify_integrity(args.original, args.restored):
        print("OK — arquivos identicos (integridade verificada).")
        log.info(f"FIM verify | resultado=OK | original='{args.original}' restored='{args.restored}'")
    else:
        sz_o = os.path.getsize(args.original)
        sz_r = os.path.getsize(args.restored)
        print(f"FALHOU — arquivos diferem! ({sz_o} bytes vs {sz_r} bytes)")
        log.warning(
            f"FIM verify | resultado=FALHOU | original='{args.original}' ({sz_o}B) "
            f"restored='{args.restored}' ({sz_r}B)"
        )
        sys.exit(1)


# =============================================================================
# CLI principal
# =============================================================================

def main():
    log = _setup_logger()

    ap = argparse.ArgumentParser(
        prog="main.py",
        description="PPM-C Compressor/Descompressor — ITI 2025.2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  Comprimir:       python main.py c dickens dickens_k6 --kmax 6\n"
            "  Comprimir+graf:  python main.py c dickens dickens_k6 --kmax 6 --graph\n"
            "  Descomprimir:    python main.py d dickens_k6 dickens_out --verify dickens\n"
            "  Benchmark:       python main.py bench dickens --kmax-max 10 --csv res.csv\n"
            "  Concatenar:      python main.py concat silesia.bin --files ../silesia/*\n"
            "  Verificar:       python main.py verify dickens dickens_out\n"
        ),
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ── c: comprimir ──────────────────────────────────────────────────────────
    p_c = sub.add_parser("c", help="Comprimir arquivo (opcionalmente gera gráfico)")
    p_c.add_argument("in_path",  help="Arquivo de entrada")
    p_c.add_argument("out_path", help="Arquivo de saida comprimido")
    p_c.add_argument("--kmax",     type=int,   required=True, help="Ordem maxima do contexto")
    p_c.add_argument("--rho",      type=int,   default=1,     help="Peso da massa de escape (padrao: 1)")
    p_c.add_argument("--j",        type=int,   default=1000,  help="Tamanho da janela de monitoramento (padrao: 1000)")
    p_c.add_argument("--pct",      type=float, default=0.30,  help="Limiar de degradacao para reset (padrao: 0.30)")
    p_c.add_argument("--no-reset", action="store_true",       help="Desativa reinicializacao adaptativa")
    p_c.add_argument("--verify",   action="store_true",       help="Verifica integridade apos compressao")
    p_c.add_argument("--graph",    action="store_true",       help="Gera grafico de comprimento medio progressivo")
    p_c.add_argument("--stride",   type=int,   default=5000,  help="Amostrar a cada N simbolos para grafico (padrao: 5000)")
    p_c.add_argument("--manifest", default=None,
                     help="Manifesto JSON (marca limites de arquivo no grafico)")

    # ── d: descomprimir ───────────────────────────────────────────────────────
    p_d = sub.add_parser("d", help="Descomprimir arquivo")
    p_d.add_argument("in_path",  help="Arquivo comprimido")
    p_d.add_argument("out_path", help="Arquivo de saida restaurado")
    p_d.add_argument("--verify", metavar="ORIGINAL", default=None,
                     help="Caminho do original para verificacao de integridade")

    # ── bench: benchmark kmax 0..N ────────────────────────────────────────────
    p_b = sub.add_parser("bench", help="Benchmark: kmax 0 a kmax-max, tabula bps e tempos")
    p_b.add_argument("in_path",              help="Arquivo a comprimir")
    p_b.add_argument("--kmax-max", type=int,   default=10,   help="Kmax maximo (padrao: 10)")
    p_b.add_argument("--rho",      type=int,   default=1)
    p_b.add_argument("--j",        type=int,   default=1000)
    p_b.add_argument("--pct",      type=float, default=0.30)
    p_b.add_argument("--no-reset", action="store_true", help="Desativa reset em todos os testes")
    p_b.add_argument("--csv",      default=None, help="Salvar tabela em CSV")

    # ── concat: concatenar arquivos ───────────────────────────────────────────
    p_cat = sub.add_parser("concat", help="Concatenar arquivos em stream unico + manifesto JSON")
    p_cat.add_argument("out_path",              help="Arquivo de saida concatenado")
    p_cat.add_argument("--files", nargs="+", required=True, help="Arquivos a concatenar")

    # ── verify: verificar integridade ─────────────────────────────────────────
    p_v = sub.add_parser("verify", help="Verificar integridade por comparacao binaria")
    p_v.add_argument("original", help="Arquivo original")
    p_v.add_argument("restored", help="Arquivo descomprimido")

    args = ap.parse_args()

    # ── Despacho de subcomandos ───────────────────────────────────────────────

    if args.cmd == "c":
        j_val = 0 if args.no_reset else args.j
        sample_stride = args.stride if args.graph else 0

        log.info(
            f"INICIO compress | in='{args.in_path}' out='{args.out_path}' "
            f"kmax={args.kmax} rho={args.rho} j={j_val} pct={args.pct:.0%} "
            f"graph={args.graph} stride={sample_stride}"
        )
        try:
            dt, n_bytes, n_bits, resets, progress = compress_file(
                args.in_path, args.out_path,
                kmax=args.kmax, rho=args.rho, j=j_val, pct=args.pct,
                sample_stride=sample_stride,
            )
        except (FileNotFoundError, ValueError) as e:
            log.error(f"ERRO compress | in='{args.in_path}' | {e}")
            sys.exit(f"Erro: {e}")

        bps    = n_bits / n_bytes if n_bytes else 0.0
        ratio  = (n_bytes * 8) / n_bits if n_bits else 0.0
        out_sz = os.path.getsize(args.out_path)
        log.info(
            f"FIM compress | in='{args.in_path}' out='{args.out_path}' | "
            f"bytes_orig={n_bytes} bits={n_bits} bps={bps:.6f} "
            f"ratio={ratio:.4f}:1 resets={len(resets)} time={dt:.3f}s"
        )
        print(
            f"\nCompressao concluida\n"
            f"  Entrada    : {_fmt_size(n_bytes)} ({n_bytes:,} bytes)\n"
            f"  Saida      : {_fmt_size(out_sz)} ({out_sz:,} bytes)\n"
            f"  bps        : {bps:.6f}\n"
            f"  Ratio      : {ratio:.4f}:1\n"
            f"  Resets     : {len(resets)}\n"
            f"  Tempo      : {dt:.3f}s"
        )

        # Gerar gráfico se --graph foi passado
        if args.graph:
            if not progress:
                sys.stderr.write("Aviso: sem dados progressivos para gerar grafico.\n")
            else:
                graph_path = args.out_path + "_graph.png"
                generate_graph(
                    progress=progress,
                    resets=resets,
                    bps_final=bps,
                    in_basename=os.path.basename(args.in_path),
                    kmax=args.kmax,
                    rho=args.rho,
                    j=j_val,
                    pct=args.pct,
                    out_png=graph_path,
                    manifest_path=args.manifest,
                )
                print(f"  Grafico    : {graph_path}")
                log.info(f"compress --graph | grafico salvo em '{graph_path}'")

        # Verificar integridade se --verify foi passado
        if args.verify:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".dec", delete=False) as tf:
                dec_path = tf.name
            try:
                decompress_file(args.out_path, dec_path, verbose=False)
                ok = verify_integrity(args.in_path, dec_path)
                print(f"  Integridade: {'OK' if ok else 'FALHOU'}")
                log.info(f"compress --verify | integridade={'OK' if ok else 'FALHOU'}")
                if not ok:
                    sys.exit(1)
            finally:
                try:
                    os.unlink(dec_path)
                except OSError:
                    pass

    elif args.cmd == "d":
        log.info(f"INICIO decompress | in='{args.in_path}' out='{args.out_path}'")
        try:
            dt, n_bytes = decompress_file(args.in_path, args.out_path)
        except (FileNotFoundError, ValueError) as e:
            log.error(f"ERRO decompress | in='{args.in_path}' | {e}")
            sys.exit(f"Erro: {e}")

        out_sz = os.path.getsize(args.out_path)
        log.info(
            f"FIM decompress | in='{args.in_path}' out='{args.out_path}' | "
            f"bytes={n_bytes} time={dt:.3f}s"
        )
        print(
            f"\nDescompressao concluida\n"
            f"  Restaurado : {_fmt_size(out_sz)} ({out_sz:,} bytes)\n"
            f"  Tempo      : {dt:.3f}s"
        )
        if args.verify:
            ok = verify_integrity(args.verify, args.out_path)
            print(f"  Integridade: {'OK' if ok else 'FALHOU'}")
            log.info(
                f"decompress --verify | original='{args.verify}' "
                f"integridade={'OK' if ok else 'FALHOU'}"
            )
            if not ok:
                sys.exit(1)

    elif args.cmd == "bench":
        cmd_bench(args, log)

    elif args.cmd == "concat":
        cmd_concat(args, log)

    elif args.cmd == "verify":
        cmd_verify(args, log)


if __name__ == "__main__":
    main()
