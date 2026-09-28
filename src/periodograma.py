import numpy as np
from astropy.timeseries import LombScargle
from scipy.stats import chi2


def extrair_segmentos_gti(time, rate, err=None, limite_gap_seg=600, min_pontos=30):
    
    dt = np.diff(time)
    quebras = np.where(dt > limite_gap_seg)[0]
    
    indices_inicio = np.insert(quebras + 1, 0, 0)
    indices_fim = np.append(quebras, len(time) - 1)
    
    segmentos = []
    for start, end in zip(indices_inicio, indices_fim):
        t_seg = time[start:end + 1]
        r_seg = rate[start:end + 1]
        e_seg = err[start:end + 1] if err is not None else None
        
        mask_seg = ~np.isnan(t_seg) & ~np.isnan(r_seg) & (r_seg >= 0)
        if e_seg is not None:
            mask_seg = mask_seg & ~np.isnan(e_seg) & (e_seg > 0)
            
        if np.sum(mask_seg) >= min_pontos:
            segmentos.append((
                t_seg[mask_seg], 
                r_seg[mask_seg], 
                e_seg[mask_seg] if e_seg is not None else None
            ))
            
    return segmentos


def analisar_abordagem_a_global(
    time, rate, error=None, 
    min_p_seg=2.0, max_p_seg=120.0, 
    n_freqs=1000, faps=[0.01, 0.001]
):
    
    mask = ~np.isnan(time) & ~np.isnan(rate)
    if error is not None:
        mask &= ~np.isnan(error) & (error > 0)
        
    t_val, r_val = time[mask], rate[mask]
    e_val = error[mask] if error is not None else None

    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)

    ls = LombScargle(t_val, r_val, e_val, normalization='standard')
    power = ls.power(freq_grid)

    idx_pico = np.argmax(power)
    best_freq = freq_grid[idx_pico]
    best_p_seg = 1.0 / best_freq

    limiares_fap = {}
    for fap in faps:
        z_crit = ls.false_alarm_level(fap, method='baluev', maximum_frequency=f_max)
        limiares_fap[fap] = float(z_crit)

    return {
        'freq_grid': freq_grid,
        'power': power,
        'best_freq': best_freq,
        'best_p_seg': best_p_seg,
        'best_power': power[idx_pico],
        'limiares_fap': limiares_fap
    }


def analisar_abordagem_b_medio(
    time,
    rate,
    error=None,
    limite_gap_seg=600,
    min_p_seg=2.0,
    max_p_seg=120.0,
    n_freqs=1000,
    min_pontos=30,
    faps=[0.01, 0.001],
):
    segmentos = extrair_segmentos_gti(
        time, rate, error, limite_gap_seg=limite_gap_seg, min_pontos=min_pontos
    )

    M = len(segmentos)
    if M == 0:
        raise ValueError("Nenhum GTI válido foi encontrado.")

    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)

    powers_list = []
    duracoes_gti = []
    info_gtis = []  

    for i, (t_seg, r_seg, e_seg) in enumerate(segmentos):
        if np.std(r_seg) == 0:
            continue

        ls = LombScargle(t_seg, r_seg, e_seg, normalization="standard")
        power = ls.power(freq_grid)

        if not np.isnan(power).any():
            powers_list.append(power)
            duracoes_gti.append(t_seg[-1] - t_seg[0])

            info_gtis.append({
                "GTI": i + 1,
                "T_Start": t_seg[0],
                "T_Stop": t_seg[-1],
                "Duracao_min": round((t_seg[-1] - t_seg[0]) / 60.0, 2),
                "N_Pontos": len(t_seg),
            })

    M_efetivo = len(powers_list)
    power_medio = np.nanmean(powers_list, axis=0)

    idx_pico = np.argmax(power_medio)
    best_freq = freq_grid[idx_pico]
    best_p_seg = 1.0 / best_freq

    duracao_media_gti = np.mean(duracoes_gti)
    n_ind = max(1.0, (f_max - f_min) * duracao_media_gti)

    limiares_fap = {}
    for fap in faps:
        p_singular = 1.0 - (1.0 - fap) ** (1.0 / n_ind)
        val_chi2 = chi2.ppf(1.0 - p_singular, df=2 * M_efetivo)
        limiares_fap[fap] = val_chi2 / (2.0 * M_efetivo)

    return {
        "freq_grid": freq_grid,
        "power_medio": power_medio,
        "best_freq": best_freq,
        "best_p_seg": best_p_seg,
        "best_power": power_medio[idx_pico],
        "n_segmentos": M_efetivo,
        "limiares_fap": limiares_fap,
        "info_gtis": info_gtis, 
    }


def funcao_janela(time, rate=None, min_p_seg=20.0, max_p_seg=120.0, n_freqs=10000):
   
    mask = ~np.isnan(time)
    if rate is not None:
        mask &= ~np.isnan(rate)
        
    t_val = time[mask]
    fake_rate = np.ones_like(t_val)
    
    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)
    
    ls = LombScargle(t_val, fake_rate, fit_mean=False, center_data=False, normalization='psd')
    power = ls.power(freq_grid)
    
    return freq_grid, power / np.max(power)


def periodograma_dinamico(
    time, rate, error=None, 
    limite_gap_seg=600, min_p_seg=20.0, max_p_seg=120.0, 
    n_freqs=10000, min_pontos=30
):
    
    segmentos = extrair_segmentos_gti(
        time, rate, error, limite_gap_seg=limite_gap_seg, min_pontos=min_pontos
    )
    
    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)
    
    matriz_potencia = []
    gti_indices = []
    
    for i, (t_seg, r_seg, e_seg) in enumerate(segmentos):
        ls = LombScargle(t_seg, r_seg, e_seg, normalization='standard')
        power = ls.power(freq_grid)
        power = np.nan_to_num(power, nan=0.0)
        
        matriz_potencia.append(power)
        gti_indices.append(i + 1)
        
    matriz_2d = np.array(matriz_potencia).T
    
    return freq_grid, gti_indices, matriz_2d