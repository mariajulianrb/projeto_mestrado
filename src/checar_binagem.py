from pathlib import Path
from astropy.table import Table
import numpy as np
import pandas as pd


def checar_binagem_pasta(caminho_pasta):
    pasta = Path(caminho_pasta)
    arquivos = sorted(pasta.glob("*.lc"))

    if not arquivos:
        print(f"Nenhum arquivo .lc encontrado em: {pasta}")
        return

    resultados = []
    for arq in arquivos:
        try:
            time = Table.read(str(arq), hdu=1)["TIME"]
            dt_med = np.median(np.diff(time))

            resultados.append(
                {
                    "Arquivo": arq.name,
                    "Binagem Medida (s)": round(float(dt_med), 2),
                    "Pontos": len(time),
                }
            )
        except Exception:
            resultados.append(
                {
                    "Arquivo": arq.name,
                    "Binagem Medida (s)": "Erro na leitura",
                    "Pontos": 0,
                }
            )

    df_resumo = pd.DataFrame(resultados)

    print(df_resumo.to_string(index=False))
    return df_resumo


if __name__ == "__main__":
    pasta_alvo = "/home/maju/projeto/LS5039/data/raw/light_curves/2017"
    checar_binagem_pasta(pasta_alvo)