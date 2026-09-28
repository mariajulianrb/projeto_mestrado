from pathlib import Path
from astropy.table import Table
import numpy as np


def relatorio_observacao(
    caminho_ficheiro,
    limite_gap_seg=900,
    t0=57629.250,
    p=3.90603,
):
    caminho = Path(caminho_ficheiro)
    tabela = Table.read(str(caminho), hdu=1)
    header = tabela.meta

    t_start = header.get("TSTART", tabela["TIME"][0])
    t_stop = header.get("TSTOP", tabela["TIME"][-1])
    exposicao_seg = header.get("EXPOSURE", 0)

    if "MJDREF" in header:
        mjd_base = float(header["MJDREF"])
    elif "MJDREFI" in header and "MJDREFF" in header:
        mjd_base = float(header["MJDREFI"]) + float(header["MJDREFF"])
    else:
        mjd_base = 55197.00076601852 

    mjd_start = mjd_base + (t_start / 86400.0)
    mjd_stop = mjd_base + (t_stop / 86400.0)

    fase_start = ((mjd_start - t0) / p) % 1.0
    fase_stop = ((mjd_stop - t0) / p) % 1.0

    duracao_total_seg = t_stop - t_start
    duracao_total_dias = duracao_total_seg / 86400.0

    periodo_nustar_seg = 96.5 * 60  # ~5790 s
    periodo_ls5039_dias = p

    orbitas_nustar_teoricas = duracao_total_seg / periodo_nustar_seg
    orbitas_ls5039 = duracao_total_dias / periodo_ls5039_dias

    time_array = np.asarray(tabela["TIME"], dtype=float)
    dt = np.diff(time_array)
    gaps = np.where(dt > limite_gap_seg)[0]
    n_janelas_reais = len(gaps) + 1

    print(f"--- Relatório HEADER: {caminho.name} ---")
    print(f"MJD Base (MJDREF)  : {mjd_base:.6f}")
    print(f"Intervalo MJD da Observação: {mjd_start:.3f} a {mjd_stop:.3f}")
    print(
        f"Cobertura de Fase Orbital : phi = {fase_start:.2f} ate phi ="
        f" {fase_stop:.2f}"
    )
    print(
        f"Duração Total : {duracao_total_dias:.2f} dias"
        f" ({duracao_total_seg:.0f} s)"
    )
    print(
        f"Tempo de Exposição Efetivo: {exposicao_seg / 1000:.2f} ks"
        f" ({exposicao_seg:.0f} s)"
    )
    print(
        f"Eficiência de Observação  :"
        f" {(exposicao_seg / duracao_total_seg) * 100:.1f}%"
    )
    print(
        "Órbitas NuSTAR :"
        f" ~{orbitas_nustar_teoricas:.1f} passagens"
    )
    print(
        "Janelas  :"
        f" {n_janelas_reais} segmentos contínuos (GTIs)"
    )
    print(
        "Órbitas do LS 5039        :"
        f" ~{orbitas_ls5039:.2f} órbitas binárias\n"
    )

    return n_janelas_reais


ficheiro = (
    "/home/maju/projeto/LS5039/data/processed/sub_2017_b10s_10_30kev.lc"
)
janelas = relatorio_observacao(ficheiro)