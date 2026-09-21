import numpy as np
from astropy.timeseries import LombScargle

def calcular_lomb_scargle(time, rate, error=None, min_p=0.05, max_p=5.0):
    """Calcula o periodograma em dias (para sinais lentos/órbitas)."""
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
    """Calcula o periodograma em Hertz (para pulsações rápidas na curva inteira)."""
    mask = (rate > 0) & ~np.isnan(rate)
    t_segundos = time[mask] 
    r_valid = rate[mask]
    e_valid = error[mask] if error is not None else None
    
    freq_min, freq_max = 1.0 / max_p_seg, 1.0 / min_p_seg  
    
    ls = LombScargle(t_segundos, r_valid, e_valid)
    freq, power = ls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max)
    
    idx_pico = np.argmax(power)
    return freq, power, freq[idx_pico], 1.0 / freq[idx_pico]

def calcular_periodograma_medio_gti(time, rate, error=None, limite_gap_seg=900, min_p_seg=2.0, max_p_seg=1800.0, n_freqs=10000):
    """
    Fatia a curva de luz nos pedaços contínuos (GTIs sem gaps), calcula o periodograma
    em Hz de cada um e devolve a MÉDIA das potências (Método de Welch / PDS).
    Especialmente indicado para busca de pulsações rápidas.
    """
    mask = (rate > 0) & ~np.isnan(rate)
    t_val = time[mask]
    r_val = rate[mask]
    e_val = error[mask] if error is not None else None
    
    # Identifica quebras (gaps orbitais)
    dt = np.diff(t_val)
    quebras = np.where(dt > limite_gap_seg)[0]
    
    indices_inicio = np.insert(quebras + 1, 0, 0)
    indices_fim = np.append(quebras, len(t_val) - 1)
    
    # Grade de frequências em Hz
    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)
    
    powers_list = []
    min_pontos = 30  # Mínimo de pontos num GTI para ser processado
    
    for i_start, i_end in zip(indices_inicio, indices_fim):
        t_seg = t_val[i_start:i_end + 1]
        r_seg = r_val[i_start:i_end + 1]
        e_seg = e_val[i_start:i_end + 1] if e_val is not None else None
        
        if len(t_seg) < min_pontos:
            continue
            
        ls = LombScargle(t_seg, r_seg, e_seg)
        power = ls.power(freq_grid)
        powers_list.append(power)
        
    if len(powers_list) == 0:
        raise ValueError("Nenhum segmento (GTI) válido foi encontrado com os parâmetros fornecidos.")
        
    # Média das potências ponto a ponto
    power_medio = np.mean(powers_list, axis=0)
    idx_pico = np.argmax(power_medio)
    best_freq = freq_grid[idx_pico]
    best_p_seg = 1.0 / best_freq
    
    n_segmentos = len(powers_list)
    print(f"Periodograma médio calculado sobre {n_segmentos} janelas orbitais contínuas (GTIs).")
    
    return freq_grid, power_medio, best_freq, best_p_seg, n_segmentos

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
    """Remove uma banda de frequências indesejada."""
    mascara_valida = (freq < f_min_excluida) | (freq > f_max_excluida)
    freq_limpa = freq[mascara_valida]
    power_limpa = power[mascara_valida]
    
    idx_pico = np.argmax(power_limpa)
    return freq_limpa, power_limpa, freq_limpa[idx_pico], 1.0 / freq_limpa[idx_pico]