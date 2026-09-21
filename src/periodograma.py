import numpy as np
from astropy.timeseries import LombScargle

def calcular_lomb_scargle(time, rate, error=None, min_p=0.05, max_p=5.0):
    """Calcula o periodograma em dias (sinais lentos/órbita)."""
    mask = (rate > 0) & ~np.isnan(rate)
    t_dias = time[mask] / 86400.0
    r_valid = rate[mask]
    e_valid = error[mask] if error is not None else None
    
    freq_min, freq_max = 1.0 / max_p, 1.0 / min_p  
    
    ls = LombScargle(t_dias, r_valid, e_valid)
    freq, power = ls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max)
    
    idx_pico = np.argmax(power)
    return freq, power, freq[idx_pico], 1.0 / freq[idx_pico]

def calcular_lomb_scargle_hz(time, rate, error=None, min_p_seg=1.0, max_p_seg=3600.0):
    """Calcula o periodograma em Hertz (alta frequência na curva inteira)."""
    mask = (rate > 0) & ~np.isnan(rate)
    t_segundos = time[mask] 
    r_valid = rate[mask]
    e_valid = error[mask] if error is not None else None
    
    freq_min, freq_max = 1.0 / max_p_seg, 1.0 / min_p_seg  
    
    ls = LombScargle(t_segundos, r_valid, e_valid)
    freq, power = ls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max)
    
    idx_pico = np.argmax(power)
    return freq, power, freq[idx_pico], 1.0 / freq[idx_pico]

def calcular_periodograma_medio_gti(
    time, rate, error=None, 
    limite_gap_seg=600,  # Padrão: 10 minutos (600s) conforme orientação
    min_p_seg=2.0, max_p_seg=1800.0, 
    n_freqs=10000, min_pontos=30
):
    """
    Fatia a curva de luz em blocos orbitais (GTIs) considerando lacunas maiores que
    `limite_gap_seg`. Calcula o PDS médio (Welch-like) e extrai metadados dos blocos.
    """
    # 1. Identifica quebras temporais na grade temporal original
    dt = np.diff(time)
    quebras = np.where(dt > limite_gap_seg)[0]
    
    indices_inicio = np.insert(quebras + 1, 0, 0)
    indices_fim = np.append(quebras, len(time) - 1)
    
    # 2. Grade comum de frequências em Hz
    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)
    
    powers_list = []
    info_gtis = []
    gti_count = 1
    
    # 3. Processa cada segmento contínuo
    for start, end in zip(indices_inicio, indices_fim):
        t_seg = time[start:end + 1]
        r_seg = rate[start:end + 1]
        e_seg = error[start:end + 1] if error is not None else None
        
        # Filtra pontos válidos dentro do bloco
        mask_seg = (r_seg > 0) & ~np.isnan(r_seg)
        
        if np.sum(mask_seg) < min_pontos:
            continue
            
        t_val = t_seg[mask_seg]
        r_val = r_seg[mask_seg]
        e_val = e_seg[mask_seg] if e_seg is not None else None
        
        # Registra metadados do bloco
        info_gtis.append({
            'GTI': gti_count,
            'T_Start': t_val[0],
            'T_Stop': t_val[-1],
            'Duracao_min': round((t_val[-1] - t_val[0]) / 60.0, 2),
            'N_Pontos': len(t_val)
        })
        gti_count += 1
        
        # Periodograma do bloco
        ls = LombScargle(t_val, r_val, e_val)
        power = ls.power(freq_grid)
        powers_list.append(power)
        
    if len(powers_list) == 0:
        raise ValueError("Nenhum segmento (GTI) válido foi encontrado com os parâmetros fornecidos.")
        
    # 4. Média das potências
    power_medio = np.mean(powers_list, axis=0)
    idx_pico = np.argmax(power_medio)
    best_freq = freq_grid[idx_pico]
    best_p_seg = 1.0 / best_freq
    n_segmentos = len(powers_list)
    
    print(f"Periodograma médio calculado sobre {n_segmentos} janelas orbitais (GTIs).")
    
    return freq_grid, power_medio, best_freq, best_p_seg, n_segmentos, info_gtis

def calcular_funcao_janela(time, min_p=0.05, max_p=5.0):
    """Calcula a Função de Janela (aliasing do satélite)."""
    mask = ~np.isnan(time)
    t_dias = time[mask] / 86400.0
    fake_rate = np.ones_like(t_dias)
    freq_min, freq_max = 1.0 / max_p, 1.0 / min_p  
    
    ls = LombScargle(t_dias, fake_rate, fit_mean=False, center_data=False)
    freq, power = ls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max)
    return freq, power / np.max(power)

def filtrar_periodograma(freq, power, f_min_excluida, f_max_excluida):
    """Remove uma banda de frequências indesejada do espectro."""
    mascara_valida = (freq < f_min_excluida) | (freq > f_max_excluida)
    freq_limpa = freq[mascara_valida]
    power_limpa = power[mascara_valida]
    
    idx_pico = np.argmax(power_limpa)
    return freq_limpa, power_limpa, freq_limpa[idx_pico], 1.0 / freq_limpa[idx_pico]