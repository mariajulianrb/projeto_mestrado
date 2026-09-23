import pandas as pd
from pathlib import Path
from astropy.io import fits

def checar_correcoes_fits(diretorio_dados):
    pasta = Path(diretorio_dados)
    arquivos = sorted(list(pasta.glob("*.lc")))
    
    resultados = []
    
    for arq in arquivos:
        with fits.open(arq) as hdul:
            header = hdul[1].header
            
            timesys = header.get('TIMESYS', 'N/A')
            barycorr_kw = header.get('BARYCORR', header.get('BARY_COR', 'N/A'))
            
            history_list = [str(h) for h in header.get('HISTORY', [])]
            history_text = " ".join(history_list).lower()
            
            tem_bary_history = 'barycorr' in history_text or 'barycentric' in history_text
            
            if timesys == 'TDB' or barycorr_kw in [True, 'YES', 'T', 1] or tem_bary_history:
                status_bary = " OK (TDB / Aplicado)"
            else:
                status_bary = "❌ NÃO APLICADO (TT/UTC)"
                
            saa_info = "Não identificado"
            if 'saamode' in history_text:
                for line in history_list:
                    if 'saamode' in line.lower():
                        saa_info = line.strip()
                        break
            elif 'saacorr' in arq.name.lower() or 'saacorr' in str(arq).lower():
                saa_info = "Detectado via caminho/nome do arquivo"

            resultados.append({
                'Arquivo': arq.name,
                'TIMESYS': timesys,
                'Correção Baricêntrica': status_bary,
                'Status SAA': saa_info
            })
            
    df = pd.DataFrame(resultados)
    return df

df_check = checar_correcoes_fits("/home/maju/projeto/LS5039/data/raw/light_curves/2017")
print(df_check.to_string())