import numpy as np
from astropy.timeseries import LombScargle


def extrair_segmentos_gti(time, rate, err=None, limite_gap_seg=600, min_pontos=30):
    """
    Fatia a curva de luz em blocos orbitais (GTIs) com base nas lacunas temporais.
    
    Retorna:
        lista de tuplas: [(t_seg, r_seg, e_seg), ...]
    """
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
        if np.sum(mask_seg) >= min_pontos:
            segmentos.append((t_seg[mask_seg], r_seg[mask_seg], e_seg[mask_seg] if e_seg is not None else None))
            
    return segmentos

def calcular_periodograma_medio_gti(
    time, rate, error=None, 
    limite_gap_seg=600,      
    min_p_seg=20.0,         
    max_p_seg=120.0,         
    n_freqs=10000, 
    min_pontos=30
): 
    """
    Calcula o PDS médio dividindo a curva em GTIs e avaliando todos
    em uma grade unificada de frequências.
    """
    dt = np.diff(time)
    quebras = np.where(dt > limite_gap_seg)[0]
    
    indices_inicio = np.insert(quebras + 1, 0, 0)
    indices_fim = np.append(quebras, len(time) - 1)
    
    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)
    
    powers_list = []
    info_gtis = []
    gti_count = 1
    
    for start, end in zip(indices_inicio, indices_fim):
        t_seg = time[start:end + 1]
        r_seg = rate[start:end + 1]
        e_seg = error[start:end + 1] if error is not None else None
        
        mask_seg = ~np.isnan(t_seg) & ~np.isnan(r_seg) & (r_seg >= 0)
        
        if e_seg is not None:
            mask_seg = mask_seg & ~np.isnan(e_seg) & (e_seg > 0)
        
        if np.sum(mask_seg) < min_pontos:
            continue
            
        t_val = t_seg[mask_seg]
        r_val = r_seg[mask_seg]
        e_val = e_seg[mask_seg] if e_seg is not None else None

        if np.std(r_val) == 0:
            continue

        ls = LombScargle(t_val, r_val, e_val)
        power = ls.power(freq_grid)
        
        if not np.isnan(power).any():
            powers_list.append(power)
            info_gtis.append({
                'GTI': gti_count,
                'T_Start': t_val[0],
                'T_Stop': t_val[-1],
                'Duracao_min': round((t_val[-1] - t_val[0]) / 60.0, 2),
                'N_Pontos': len(t_val)
            })
            gti_count += 1
        
    if len(powers_list) == 0:
        raise ValueError("Nenhum segmento (GTI) válido gerou um espectro sem NaNs.")
        
    power_medio = np.nanmean(powers_list, axis=0)
    idx_pico = np.argmax(power_medio)
    best_freq = freq_grid[idx_pico]
    best_p_seg = 1.0 / best_freq
    n_segmentos = len(powers_list)
    
    print(f"Periodograma médio calculado sobre {n_segmentos} janelas orbitais (GTIs).")
    
    return freq_grid, power_medio, best_freq, best_p_seg, n_segmentos, info_gtis


def calcular_limiares_fap_scargle(freq_grid, faps=[0.01, 0.001, 0.0001]):
    """
    Calcula os limiares analíticos de FAP segundo Scargle (1982).
    
    Fórmula: z_crit = ln(N_ind) - ln(FAP)
    """
    df = np.median(np.diff(freq_grid))
    f_max = np.max(freq_grid)
    
    # Número aproximado de frequências independentes (N_ind)
    n_ind = f_max / df
    
    limiares = {}
    for fap in faps:
        z_crit = np.log(n_ind) - np.log(fap)
        limiares[fap] = z_crit
        
    return limiares, n_ind


def calcular_funcao_janela(
    time, rate=None, 
    min_p_seg=20.0, max_p_seg=120.0, 
    n_freqs=10000
):
    """Calcula a Função de Janela (aliasing orbital) na mesma grade de frequências."""
    if rate is not None:
        mask = ~np.isnan(time) & ~np.isnan(rate)
    else:
        mask = ~np.isnan(time)
        
    t_val = time[mask]
    fake_rate = np.ones_like(t_val)
    
    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)
    
    ls = LombScargle(t_val, fake_rate, fit_mean=False, center_data=False)
    power = ls.power(freq_grid)
    
    return freq_grid, power / np.max(power)


def filtrar_periodograma(freq, power, f_min_excluida, f_max_excluida):
    """Remove uma banda de frequências indesejada e recalcula o pico principal."""
    mascara_valida = (freq < f_min_excluida) | (freq > f_max_excluida)
    freq_limpa = freq[mascara_valida]
    power_limpa = power[mascara_valida]
    
    idx_pico = np.argmax(power_limpa)
    return freq_limpa, power_limpa, freq_limpa[idx_pico], 1.0 / freq_limpa[idx_pico]

def calcular_periodograma_dinamico(
    time, rate, error=None, 
    limite_gap_seg=600, min_p_seg=20.0, max_p_seg=120.0, 
    n_freqs=10000, min_pontos=30
):
    """
    Gera uma matriz 2D de potência (Frequência x GTI) para visualização dinâmica.
    
    Retorna:
        freq_grid: array 1D com as frequências
        gti_indices: lista com os números dos GTIs
        matriz_2d: array 2D de formato (n_freqs, n_gtis)
    """
    segmentos = extrair_segmentos_gti(
        time, rate, error, limite_gap_seg=limite_gap_seg, min_pontos=min_pontos
    )
    
    f_min, f_max = 1.0 / max_p_seg, 1.0 / min_p_seg
    freq_grid = np.linspace(f_min, f_max, n_freqs)
    
    matriz_potencia = []
    gti_indices = []
    
    for i, (t_seg, r_seg, e_seg) in enumerate(segmentos):
        ls = LombScargle(t_seg, r_seg, e_seg)
        power = ls.power(freq_grid)
        
        
        power = np.nan_to_num(power, nan=0.0)
        
        matriz_potencia.append(power)
        gti_indices.append(i + 1)
        
   
    matriz_2d = np.array(matriz_potencia).T
    
    return freq_grid, gti_indices, matriz_2d