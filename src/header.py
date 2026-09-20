from astropy.table import Table

def relatorio_observacao(caminho_ficheiro):
    tabela = Table.read(caminho_ficheiro, hdu=1)
    header = tabela.meta
    
    t_start = header.get('TSTART', tabela['TIME'][0])
    t_stop = header.get('TSTOP', tabela['TIME'][-1])
    
    exposicao_seg = header.get('EXPOSURE', 0)
    
    duracao_total_seg = t_stop - t_start
    duracao_total_dias = duracao_total_seg / 86400.0
    
    periodo_nustar_seg = 96.5 * 60      
    periodo_ls5039_dias = 3.90603       

    orbitas_nustar = duracao_total_seg / periodo_nustar_seg
    orbitas_ls5039 = duracao_total_dias / periodo_ls5039_dias
    
    print(f"--- Relatório do Ficheiro: {caminho_ficheiro.split('/')[-1]} ---")
    print(f"Duração Total da Campanha : {duracao_total_dias:.2f} dias ({duracao_total_seg:.0f} segundos)")
    print(f"Tempo de Exposição Efetivo: {exposicao_seg / 1000:.2f} ks ({exposicao_seg:.0f} segundos)")
    print(f"Órbitas do Satélite NuSTAR: ~{orbitas_nustar:.1f} órbitas da Terra")
    print(f"Órbitas do Sistema LS 5039: ~{orbitas_ls5039:.2f} órbitas binárias\n")

ficheiro_2017 = "/home/maju/projeto/LS5039/data/raw/light_curves/2017/nu30201034002A01_2017_b1h_10_30kev_src.lc"

relatorio_observacao(ficheiro_2017)
# relatorio_observacao(ficheiro_2022)