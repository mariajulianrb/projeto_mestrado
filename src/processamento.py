import numpy as np
from astropy.table import Table

def subtrair_fundo(arquivo_src, arquivo_bkg, raio_src, raio_bkg, arquivo_saida):
    """
    Parâmetros:
    - arquivo_src (str): Caminho para o ficheiro FITS da fonte.
    - arquivo_bkg (str): Caminho para o ficheiro FITS do fundo.
    - raio_src (float): Raio da região de extração da fonte (em píxeis físicos do ds9).
    - raio_bkg (float): Raio da região de extração do fundo.
    - arquivo_saida (str): Caminho e nome do ficheiro FITS final a ser salvo.
    """
    
    src = Table.read(arquivo_src, hdu=1)
    bkg = Table.read(arquivo_bkg, hdu=1)

    a = (raio_src ** 2) / (raio_bkg ** 2)
    
    print(f"Fator de escala de área (a) calculado: {a:.4f}")

    rate_sub = src['RATE'] - (a * bkg['RATE'])
    err_sub = np.sqrt(src['ERROR'] ** 2 + (a * bkg['ERROR']) ** 2)

    lc_clean = src.copy()
    lc_clean['RATE'] = rate_sub
    lc_clean['ERROR'] = err_sub

    lc_clean.write(arquivo_saida, format='fits', overwrite=True)
    print(f"Sucesso! Ficheiro corrigido salvo em: {arquivo_saida}\n")