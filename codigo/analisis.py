#!/usr/bin/env python3
"""
Análisis estadístico y generación de gráficas/tablas para el Taller MPI.

Lee los .dat producidos por lanzador.pl en la carpeta Soluciones/ y produce:
  - resumen.csv: tabla con N, hilos, np, algoritmo, n, media, std, speedup, eficiencia
  - tablas pareadas FxC vs FxT (por np)
  - 5 gráficas por algoritmo (tiempo, speedup, eficiencia, barras, boxplot)
  - gráficas comparativas FxC vs FxT
  - (opcional) comparación entre sistemas si se pasan varias carpetas

Uso:
    python analisis.py                       # usa carpeta Soluciones/
    python analisis.py Soluciones_cluster Soluciones_local1 Soluciones_local2
"""

import os
import re
import sys
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PATRON = re.compile(
    r"(mxmOmpMPI(?:fxc|fxt))-(\d+)-np_(\d+)-Hilos-(\d+)\.dat$"
)


def cargar_carpeta(carpeta, etiqueta_sistema=None):
    """Devuelve un DataFrame con (sistema, algoritmo, N, np, hilos, tiempos)."""
    filas = []
    if not os.path.isdir(carpeta):
        print(f"[WARN] Carpeta no existe: {carpeta}")
        return pd.DataFrame()
    for fn in sorted(os.listdir(carpeta)):
        m = PATRON.match(fn)
        if not m:
            continue
        ejecutable, N, np_proc, hilos = m.groups()
        algoritmo = "FxC" if ejecutable.endswith("fxc") else "FxT"
        ruta = os.path.join(carpeta, fn)
        with open(ruta) as f:
            tiempos_us = []
            for linea in f:
                linea = linea.strip()
                if not linea:
                    continue
                try:
                    tiempos_us.append(float(linea))
                except ValueError:
                    pass
        if not tiempos_us:
            continue
        tiempos_s = np.array(tiempos_us) / 1e6
        filas.append({
            "sistema": etiqueta_sistema or os.path.basename(carpeta.rstrip("/\\")),
            "algoritmo": algoritmo,
            "N": int(N),
            "np": int(np_proc),
            "hilos": int(hilos),
            "n": len(tiempos_s),
            "media_s": float(np.mean(tiempos_s)),
            "std_s": float(np.std(tiempos_s, ddof=1)) if len(tiempos_s) > 1 else 0.0,
            "min_s": float(np.min(tiempos_s)),
            "max_s": float(np.max(tiempos_s)),
            "tiempos_s": tiempos_s,
        })
    return pd.DataFrame(filas)


def calcular_speedup_eficiencia(df):
    """Añade columnas speedup y eficiencia respecto al caso de 1 hilo."""
    df = df.copy()
    df["speedup"] = np.nan
    df["eficiencia"] = np.nan
    grupos = df.groupby(["sistema", "algoritmo", "N", "np"])
    for clave, sub in grupos:
        base = sub[sub["hilos"] == 1]
        if base.empty:
            continue
        t1 = base["media_s"].iloc[0]
        for idx in sub.index:
            t = df.at[idx, "media_s"]
            h = df.at[idx, "hilos"]
            df.at[idx, "speedup"] = t1 / t if t > 0 else np.nan
            df.at[idx, "eficiencia"] = (t1 / t) / h if t > 0 and h > 0 else np.nan
    return df


def fig_tiempo_vs_hilos(df, algoritmo, np_proc, salida):
    sub = df[(df["algoritmo"] == algoritmo) & (df["np"] == np_proc)]
    if sub.empty:
        return
    plt.figure(figsize=(7, 4.5))
    for N, grupo in sub.groupby("N"):
        grupo = grupo.sort_values("hilos")
        plt.errorbar(grupo["hilos"], grupo["media_s"], yerr=grupo["std_s"],
                     marker="o", label=f"N = {N}", capsize=3)
    plt.yscale("log")
    plt.xlabel("Número de hilos OpenMP")
    plt.ylabel("Tiempo medio de ejecución (s, escala log)")
    plt.title(f"Tiempo medio de ejecución vs. número de hilos\nAlgoritmo {algoritmo}, np = {np_proc}")
    plt.legend(title="Tamaño matriz")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida, dpi=130)
    plt.close()


def fig_speedup(df, algoritmo, np_proc, salida):
    sub = df[(df["algoritmo"] == algoritmo) & (df["np"] == np_proc)]
    if sub.empty:
        return
    plt.figure(figsize=(7, 4.5))
    hilos_max = int(sub["hilos"].max())
    ideal = list(range(1, hilos_max + 1))
    plt.plot(ideal, ideal, "--", color="gray", label="Speedup ideal (lineal)")
    for N, grupo in sub.groupby("N"):
        grupo = grupo.sort_values("hilos")
        plt.plot(grupo["hilos"], grupo["speedup"], marker="o", label=f"N = {N}")
    plt.xlabel("Número de hilos OpenMP")
    plt.ylabel("Speedup S(p) = T(1) / T(p)")
    plt.title(f"Speedup observado vs. ideal\nAlgoritmo {algoritmo}, np = {np_proc}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida, dpi=130)
    plt.close()


def fig_eficiencia(df, algoritmo, np_proc, salida):
    sub = df[(df["algoritmo"] == algoritmo) & (df["np"] == np_proc)]
    if sub.empty:
        return
    plt.figure(figsize=(7, 4.5))
    plt.axhline(1.0, ls="--", color="gray", label="Eficiencia ideal = 1.0")
    for N, grupo in sub.groupby("N"):
        grupo = grupo.sort_values("hilos")
        plt.plot(grupo["hilos"], grupo["eficiencia"], marker="o", label=f"N = {N}")
    plt.xlabel("Número de hilos OpenMP")
    plt.ylabel("Eficiencia E(p) = S(p) / p")
    plt.title(f"Eficiencia del paralelismo\nAlgoritmo {algoritmo}, np = {np_proc}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida, dpi=130)
    plt.close()


def fig_barras(df, algoritmo, np_proc, salida):
    sub = df[(df["algoritmo"] == algoritmo) & (df["np"] == np_proc)]
    if sub.empty:
        return
    Ns = sorted(sub["N"].unique())
    hilos_unicos = sorted(sub["hilos"].unique())
    x = np.arange(len(Ns))
    ancho = 0.8 / len(hilos_unicos)
    plt.figure(figsize=(8, 4.8))
    for i, h in enumerate(hilos_unicos):
        medias = [sub[(sub["N"] == n) & (sub["hilos"] == h)]["media_s"].mean() for n in Ns]
        stds = [sub[(sub["N"] == n) & (sub["hilos"] == h)]["std_s"].mean() for n in Ns]
        plt.bar(x + i * ancho, medias, ancho, yerr=stds, capsize=3, label=f"{h} hilo(s)")
    plt.yscale("log")
    plt.xticks(x + ancho * (len(hilos_unicos) - 1) / 2, [f"N = {n}" for n in Ns])
    plt.xlabel("Tamaño de matriz N")
    plt.ylabel("Tiempo medio (s, escala log)")
    plt.title(f"Comparativa de tiempo por número de hilos\nAlgoritmo {algoritmo}, np = {np_proc}")
    plt.legend(title="Hilos")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(salida, dpi=130)
    plt.close()


def fig_boxplot(df, algoritmo, np_proc, salida):
    sub = df[(df["algoritmo"] == algoritmo) & (df["np"] == np_proc)]
    if sub.empty:
        return
    Ns = sorted(sub["N"].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(4 * len(Ns), 4.5), sharey=False)
    if len(Ns) == 1:
        axes = [axes]
    for ax, n in zip(axes, Ns):
        datos_por_hilo = []
        etiquetas = []
        for h in sorted(sub[sub["N"] == n]["hilos"].unique()):
            tiempos = sub[(sub["N"] == n) & (sub["hilos"] == h)]["tiempos_s"].iloc[0]
            datos_por_hilo.append(tiempos)
            etiquetas.append(str(h))
        ax.boxplot(datos_por_hilo, tick_labels=etiquetas)
        ax.set_title(f"N = {n}")
        ax.set_xlabel("Hilos")
        ax.set_ylabel("Tiempo de ejecución (s)")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Distribución de tiempos por configuración\nAlgoritmo {algoritmo}, np = {np_proc}")
    plt.tight_layout()
    plt.savefig(salida, dpi=130)
    plt.close()


def fig_comparativa_fxc_fxt(df, np_proc, salida):
    sub = df[df["np"] == np_proc]
    if sub.empty or sub["algoritmo"].nunique() < 2:
        return
    Ns = sorted(sub["N"].unique())
    hilos_unicos = sorted(sub["hilos"].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(4.5 * len(Ns), 4.5), sharey=False)
    if len(Ns) == 1:
        axes = [axes]
    x = np.arange(len(hilos_unicos))
    ancho = 0.35
    for ax, n in zip(axes, Ns):
        for i, alg in enumerate(["FxC", "FxT"]):
            medias = []
            errs = []
            for h in hilos_unicos:
                fila = sub[(sub["N"] == n) & (sub["hilos"] == h) & (sub["algoritmo"] == alg)]
                medias.append(fila["media_s"].mean() if not fila.empty else 0)
                errs.append(fila["std_s"].mean() if not fila.empty else 0)
            ax.bar(x + i * ancho, medias, ancho, yerr=errs, capsize=3, label=alg)
        ax.set_xticks(x + ancho / 2)
        ax.set_xticklabels(hilos_unicos)
        ax.set_xlabel("Hilos OpenMP")
        ax.set_ylabel("Tiempo medio (s)")
        ax.set_title(f"N = {n}")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"Comparativa FxC vs FxT, np = {np_proc}")
    plt.tight_layout()
    plt.savefig(salida, dpi=130)
    plt.close()


def fig_comparativa_sistemas(df, algoritmo, np_proc, salida):
    sub = df[(df["algoritmo"] == algoritmo) & (df["np"] == np_proc)]
    if sub.empty or sub["sistema"].nunique() < 2:
        return
    Ns = sorted(sub["N"].unique())
    sistemas = sorted(sub["sistema"].unique())
    fig, axes = plt.subplots(1, len(Ns), figsize=(4.5 * len(Ns), 4.5), sharey=False)
    if len(Ns) == 1:
        axes = [axes]
    for ax, n in zip(axes, Ns):
        for sis in sistemas:
            grupo = sub[(sub["N"] == n) & (sub["sistema"] == sis)].sort_values("hilos")
            if grupo.empty:
                continue
            ax.plot(grupo["hilos"], grupo["media_s"], marker="o", label=sis)
        ax.set_xlabel("Hilos OpenMP")
        ax.set_ylabel("Tiempo medio (s)")
        ax.set_title(f"N = {n}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Comparativa entre sistemas — {algoritmo}, np = {np_proc}")
    plt.tight_layout()
    plt.savefig(salida, dpi=130)
    plt.close()


def tabla_pareada(df, np_proc, salida_csv):
    sub = df[df["np"] == np_proc]
    if sub.empty:
        return
    pivot = sub.pivot_table(
        index=["N", "hilos"],
        columns="algoritmo",
        values=["media_s", "std_s", "speedup", "eficiencia"],
        aggfunc="mean",
    )
    if "FxC" in sub["algoritmo"].values and "FxT" in sub["algoritmo"].values:
        try:
            pivot[("delta_pct", "")] = (
                (pivot[("media_s", "FxC")] - pivot[("media_s", "FxT")])
                / pivot[("media_s", "FxC")] * 100
            )
        except KeyError:
            pass
    pivot.to_csv(salida_csv)


def main():
    carpetas = sys.argv[1:] if len(sys.argv) > 1 else ["Soluciones"]

    dfs = []
    for c in carpetas:
        etiqueta = os.path.basename(c.rstrip("/\\")) or c
        d = cargar_carpeta(c, etiqueta)
        if not d.empty:
            dfs.append(d)

    if not dfs:
        print("No se encontraron archivos .dat en las carpetas indicadas.")
        return

    df = pd.concat(dfs, ignore_index=True)
    df = calcular_speedup_eficiencia(df)

    os.makedirs("Analisis", exist_ok=True)

    # CSV resumen (sin la columna de tiempos crudos)
    resumen = df.drop(columns=["tiempos_s"]).sort_values(
        ["sistema", "algoritmo", "N", "np", "hilos"]
    )
    resumen.to_csv("Analisis/resumen.csv", index=False)
    print(f"[OK] Analisis/resumen.csv ({len(resumen)} filas)")

    # Gráficas por algoritmo y np (solo del primer sistema, típicamente el cluster)
    sistema_principal = df["sistema"].iloc[0]
    df_principal = df[df["sistema"] == sistema_principal]

    for alg in df_principal["algoritmo"].unique():
        for np_proc in sorted(df_principal["np"].unique()):
            base = f"Analisis/{alg}_np{np_proc}"
            fig_tiempo_vs_hilos(df_principal, alg, np_proc, f"{base}_01_tiempo.png")
            fig_speedup(df_principal, alg, np_proc, f"{base}_02_speedup.png")
            fig_eficiencia(df_principal, alg, np_proc, f"{base}_03_eficiencia.png")
            fig_barras(df_principal, alg, np_proc, f"{base}_04_barras.png")
            fig_boxplot(df_principal, alg, np_proc, f"{base}_05_boxplot.png")
            print(f"[OK] gráficas {alg} np={np_proc}")

    # Comparativa FxC vs FxT
    for np_proc in sorted(df_principal["np"].unique()):
        fig_comparativa_fxc_fxt(df_principal, np_proc, f"Analisis/cmp_FxCvsFxT_np{np_proc}.png")
        tabla_pareada(df_principal, np_proc, f"Analisis/tabla_pareada_np{np_proc}.csv")

    # Comparativa entre sistemas (si hay más de uno)
    if df["sistema"].nunique() > 1:
        for alg in df["algoritmo"].unique():
            for np_proc in sorted(df["np"].unique()):
                fig_comparativa_sistemas(
                    df, alg, np_proc, f"Analisis/cmp_sistemas_{alg}_np{np_proc}.png"
                )
        print("[OK] gráficas comparativa entre sistemas")

    print("\n[FIN] Resultados en Analisis/")


if __name__ == "__main__":
    main()
