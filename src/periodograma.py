import numpy as np
from astropy.timeseries import LombScargle

def calcular_lomb_scargle(time, rate, error=None, min_p=0.05, max_p=5.0):
    """Calcula o periodograma em dias (para sinais lentos e órbitas)."""
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
    """Calcula o periodograma em Hertz (para pulsações rápidas)."""
    mask = (rate > 0) & ~np.isnan(rate)
    t_segundos = time[mask] 
    r_valid = rate[mask]
    e_valid = error[mask] if error is not None else None
    
    freq_min, freq_max = 1.0 / max_p_seg, 1.0 / min_p_seg  
    
    ls = LombScargle(t_segundos, r_valid, e_valid)
    freq, power = ls.autopower(minimum_frequency=freq_min, maximum_frequency=freq_max)
    
    idx_pico = np.argmax(power)
    return freq, power, freq[idx_pico], 1.0 / freq[idx_pico]

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