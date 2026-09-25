import numpy as np
import pandas as pd
from astropy.table import Table

def subtrair_fundo(arquivo_src, arquivo_bkg, raio_src, raio_bkg, arquivo_saida):

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


def calcular_fase_orbital(tabela_astropy, t0=57629.250, p=3.90603, mjd_base=55197.00076601852):

    df = tabela_astropy.to_pandas()
    
    df = df[df['RATE'] > 0].copy()
    
    df['MJD'] = mjd_base + (df['TIME'] / 86400.0)
    df['FASE'] = ((df['MJD'] - t0) / p) % 1.0
    
    df = df.sort_values(by='FASE')
    df_ciclo2 = df.copy()
    df_ciclo2['FASE'] = df_ciclo2['FASE'] + 1.0
    
    return pd.concat([df, df_ciclo2])