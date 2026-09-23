import numpy as np
import pandas as pd
from pathlib import Path
from astropy.io import fits

def auditar_pasta_curvas(diretorio_dados):
    pasta = Path(diretorio_dados)
    arquivos = sorted(list(pasta.glob("*.lc")))
    
    if not arquivos:
        print(f"Nenhum arquivo .lc encontrado em: {pasta}")
        return None

    relatorio = []

    for arq in arquivos:
        try:
            with fits.open(arq) as hdul:
                header = hdul[1].header
                data = hdul[1].data
                
                # 1. Informações de tempo e amostragem
                time = data['TIME']
                rate = data['RATE'] if 'RATE' in data.names else data['COUNTS']
                dt = np.diff(time)
                dt_validos = dt[dt > 0]
                
                timedel_hdr = header.get('TIMEDEL', header.get('DELTAT', np.nan))
                dt_mediano = np.median(dt_validos) if len(dt_validos) > 0 else np.nan
                dt_min = np.min(dt_validos) if len(dt_validos) > 0 else np.nan
                dt_max = np.max(dt_validos) if len(dt_validos) > 0 else np.nan
                
                # 2. Informações de Energia e Canal PI (se presentes no Header)
                e_min = header.get('E_MIN', header.get('CHANMIN', 'N/A'))
                e_max = header.get('E_MAX', header.get('CHANMAX', 'N/A'))
                
                Status = "OK"
                Alertas = []
                
                # Checa se o TIMEDEL do cabeçalho condiz com a amostragem real
                if not np.isnan(timedel_hdr) and not np.isclose(timedel_hdr, dt_mediano, rtol=1e-2):
                    Alertas.append(f"Header ({timedel_hdr}s) != Dado ({dt_mediano:.1f}s)")
                
                # Checa se há contaminação por tempos de fótons (passos curtos irregulares)
                if dt_min < 0.5 * dt_mediano:
                    Alertas.append(f"Micro-passos detectados (min dt = {dt_min:.1f}s)")
                
                # Checa se há NaN ou dados zerados/negativos
                nans = np.sum(np.isnan(rate))
                if nans > 0:
                    Alertas.append(f"{nans} NaNs no rate")

                if len(Alertas) > 0:
                    Status = " ATENÇÃO: " + " | ".join(Alertas)
                
                relatorio.append({
                    'Arquivo': arq.name,
                    'T_Total (ks)': round((time[-1] - time[0]) / 1000.0, 2),
                    'N_Pontos': len(time),
                    'TIMEDEL (Hdr)': timedel_hdr,
                    'dt Mediano': round(dt_mediano, 2),
                    'dt Min': round(dt_min, 2),
                    'dt Max': round(dt_max, 2),
                    'E_Min': e_min,
                    'E_Max': e_max,
                    'Status': Status
                })
        except Exception as e:
            relatorio.append({
                'Arquivo': arq.name,
                'Status': f"ERRO AO ABRIR: {str(e)}"
            })

    df_relatorio = pd.DataFrame(relatorio)
    return df_relatorio

caminho_dados = "/home/maju/projeto/LS5039/data/raw/light_curves/2017"
df_audit = auditar_pasta_curvas(caminho_dados)

print(df_audit)