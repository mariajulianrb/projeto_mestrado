import numpy as np
from pathlib import Path
from astropy.table import Table

def relatorio_observacao(caminho_ficheiro, limite_gap_seg=900):
    caminho = Path(caminho_ficheiro)
    tabela = Table.read(str(caminho), hdu=1)
    header = tabela.meta
    
    t_start = header.get('TSTART', tabela['TIME'][0])
    t_stop = header.get('TSTOP', tabela['TIME'][-1])
    exposicao_seg = header.get('EXPOSURE', 0)
    
    duracao_total_seg = t_stop - t_start
    duracao_total_dias = duracao_total_seg / 86400.0
    
    periodo_nustar_seg = 96.5 * 60       # ~5790s
    periodo_ls5039_dias = 3.90603       

    orbitas_nustar_teoricas = duracao_total_seg / periodo_nustar_seg
    orbitas_ls5039 = duracao_total_dias / periodo_ls5039_dias
    
    # contagem de janelas contínuas (GTIs)
    time_array = np.asarray(tabela['TIME'], dtype=float)
    dt = np.diff(time_array)
    
    # gaps maiores que o limite 
    gaps = np.where(dt > limite_gap_seg)[0]
    n_janelas_reais = len(gaps) + 1
    
    print(f"--- Relatório do Ficheiro: {caminho.name} ---")
    print(f"Duração Total da Campanha : {duracao_total_dias:.2f} dias ({duracao_total_seg:.0f} s)")
    print(f"Tempo de Exposição Efetivo: {exposicao_seg / 1000:.2f} ks ({exposicao_seg:.0f} s)")
    print(f"Eficiência de Observação  : {(exposicao_seg / duracao_total_seg) * 100:.1f}%")
    print(f"Órbitas NuSTAR (Teóricas) : ~{orbitas_nustar_teoricas:.1f} passagens")
    print(f"Janelas Reais de Dados    : {n_janelas_reais} segmentos contínuos (GTIs)")
    print(f"Órbitas do LS 5039        : ~{orbitas_ls5039:.2f} órbitas binárias\n")

    return n_janelas_reais

ficheiro = "/home/maju/projeto/LS5039/data/processed/sub_2017_b10s_10_30kev.lc"
janelas = relatorio_observacao(ficheiro)