"""
=============================================================================
SIMULATED ANNEALING - Ottimizzazione costi raddoppio linea Merano-Malles
=============================================================================
Obiettivo: minimizzare i km di binario da raddoppiare per permettere
gli incroci tra treni in direzioni opposte.

Variabili indipendenti: t1..t6 (orari di partenza in minuti dall'inizio ora)
  - t1: lento rosso    Merano → Malles
  - t2: lento verde    Merano → Malles
  - t3: veloce blu     Merano → Malles
  - t4: lento rosso    Malles → Merano
  - t5: lento verde    Malles → Merano
  - t6: veloce blu     Malles → Merano

Pattern: si ripete ogni 60 minuti. Simuliamo 5 ore (per catturare tutti
gli incroci nella finestra 10-13).
=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import math

# =============================================================================
# PARAMETRI ALGORITMO (modificabili facilmente)
# =============================================================================
SA_MAX_ITERATIONS  = 50_000   # numero massimo di iterazioni
SA_INITIAL_TEMP    = 100.0    # temperatura iniziale
SA_COOLING_RATE    = 0.9995   # fattore di raffreddamento (moltiplicativo)
SA_MIN_TEMP        = 0.01     # temperatura minima (stop anticipato)
SA_PERTURB_MAX_MIN = 10.0     # perturbazione massima in minuti (±N)

# Finestra di simulazione per la visualizzazione
VIZ_START_MIN = 10 * 60   # 10:00 in minuti dall'inizio giornata
VIZ_END_MIN   = 13 * 60   # 13:00 in minuti dall'inizio giornata

# Offset di partenza: i treni della "prima ora" partono tra 9:00 e 10:00
BASE_HOUR_MIN = 9 * 60    # 9:00 → in minuti dall'inizio giornata

# =============================================================================
# INFRASTRUTTURA: sezioni di blocco e stazioni
# =============================================================================
# Ogni elemento è un dict con:
#   name, abbr, km, tracks (numero binari), stop_slow, stop_fast
INFRASTRUCTURE = [
    {"name": "Meran",               "abbr": "ME",           "km": 0.000,  "tracks": 4, "stop_slow": True,  "stop_fast": True},
    {"name": "Algund",              "abbr": "LAG",          "km": 1.517,  "tracks": 1, "stop_slow": True,  "stop_fast": False},
    {"name": "SblLag02041",         "abbr": "SblLag02041",  "km": 2.041,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblLag03041",         "abbr": "SblLag03041",  "km": 3.041,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Marling",             "abbr": "MAR",          "km": 3.566,  "tracks": 2, "stop_slow": True,  "stop_fast": False},
    {"name": "SblMar04492",         "abbr": "SblMar04492",  "km": 4.492,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblMar05492",         "abbr": "SblMar05492",  "km": 5.492,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Forst",               "abbr": "vbForst",      "km": 6.017,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblMar06492",         "abbr": "SblMar06492",  "km": 6.492,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblMar07492",         "abbr": "SblMar07492",  "km": 7.492,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Josefsberg",          "abbr": "vbJos",        "km": 8.417,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblMar08492",         "abbr": "SblMar08492",  "km": 8.492,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblMar09492",         "abbr": "SblMar09492",  "km": 9.492,  "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Toel",                "abbr": "TEL",          "km": 10.419, "tracks": 2, "stop_slow": True,  "stop_fast": False},
    {"name": "SblTe111108",         "abbr": "SblTe111108",  "km": 11.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Rabland",             "abbr": "RAB",          "km": 11.699, "tracks": 1, "stop_slow": True,  "stop_fast": False},
    {"name": "SblTe112108",         "abbr": "SblTe112108",  "km": 12.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblTe113108",         "abbr": "SblTe113108",  "km": 13.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Plaus",               "abbr": "PLA",          "km": 13.991, "tracks": 1, "stop_slow": True,  "stop_fast": False},
    {"name": "SblTe114108",         "abbr": "SblTe114108",  "km": 14.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblTe115108",         "abbr": "SblTe115108",  "km": 15.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblTe116108",         "abbr": "SblTe116108",  "km": 16.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblTe117108",         "abbr": "SblTe117108",  "km": 17.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Naturns",             "abbr": "NAT",          "km": 17.277, "tracks": 1, "stop_slow": True,  "stop_fast": True},
    {"name": "SblTe118108",         "abbr": "SblTe118108",  "km": 18.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "stNaturnsKompatsch",  "abbr": "NatKomp",      "km": 18.397, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiTe119108",         "abbr": "SbiTe119108",  "km": 19.108, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Staben",              "abbr": "STAV",         "km": 19.797, "tracks": 2, "stop_slow": True,  "stop_fast": False},
    {"name": "SbiStab20529",        "abbr": "SbiStav20529", "km": 20.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiStab21529",        "abbr": "SbiStav21529", "km": 21.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiStab22529",        "abbr": "SbiStav22529", "km": 22.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Tschars",             "abbr": "CIA",          "km": 22.578, "tracks": 1, "stop_slow": True,  "stop_fast": False},
    {"name": "SbiStab23529",        "abbr": "SbiStav23529", "km": 23.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiStab24529",        "abbr": "SbiStav24529", "km": 24.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblStab25529",        "abbr": "SblStav25529", "km": 25.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Kastelbell",          "abbr": "CAB",          "km": 26.133, "tracks": 1, "stop_slow": True,  "stop_fast": False},
    {"name": "SbiStab26529",        "abbr": "SbiStav26529", "km": 26.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiStab27529",        "abbr": "SbiStav27529", "km": 27.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblStab28529",        "abbr": "SblStav28529", "km": 28.529, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Latsch",              "abbr": "LAC",          "km": 29.261, "tracks": 2, "stop_slow": True,  "stop_fast": True},
    {"name": "SbiLac30026",         "abbr": "SbiLac30026",  "km": 30.026, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiLac31026",         "abbr": "SbiLac31026",  "km": 31.026, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblLac32026",         "abbr": "SblLac32026",  "km": 32.026, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Goldrain",            "abbr": "COLD",         "km": 32.354, "tracks": 1, "stop_slow": True,  "stop_fast": False},
    {"name": "SbiLac33026",         "abbr": "SbiLac33026",  "km": 33.026, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiLac34026",         "abbr": "SbiLac34026",  "km": 34.026, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblLac35026",         "abbr": "SblLac35026",  "km": 35.026, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiLac36026",         "abbr": "SblLac36026",  "km": 36.026, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiLac37026",         "abbr": "SbiLac37026",  "km": 37.026, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Schlanders",          "abbr": "SIL",          "km": 37.792, "tracks": 2, "stop_slow": True,  "stop_fast": True},
    {"name": "SbiSi138377",         "abbr": "SbiSi138377",  "km": 38.377, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiSi139377",         "abbr": "SbiSi139377",  "km": 39.377, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblSi140377",         "abbr": "SblSi140377",  "km": 40.377, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiSi141377",         "abbr": "SbiSi141377",  "km": 41.377, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiSi142377",         "abbr": "SbiSi142377",  "km": 42.377, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblSi143377",         "abbr": "SblSi143377",  "km": 43.377, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblSi144377",         "abbr": "SblSi144377",  "km": 44.377, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Laas",                "abbr": "LASA",         "km": 44.962, "tracks": 2, "stop_slow": True,  "stop_fast": True},
    {"name": "SbiLasa45915",        "abbr": "SbiLasa45915", "km": 45.915, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblLasa46915",        "abbr": "SblLasa46915", "km": 46.915, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblLasa47915",        "abbr": "SblLasa47915", "km": 47.915, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblLasa48915",        "abbr": "SblLasa48915", "km": 48.915, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Eyrs",                "abbr": "ORIS",         "km": 49.020, "tracks": 1, "stop_slow": True,  "stop_fast": False},
    {"name": "SblLasa49915",        "abbr": "SblLasa49915", "km": 49.915, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblLasa50915",        "abbr": "SblLasa50915", "km": 50.915, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Spondinig",           "abbr": "SPON",         "km": 51.868, "tracks": 2, "stop_slow": True,  "stop_fast": False},
    {"name": "SbiSpon52860",        "abbr": "SbiSpon52860", "km": 52.860, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblSpon53860",        "abbr": "SblSpon53860", "km": 53.860, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblSpon54860",        "abbr": "SblSpon54860", "km": 54.860, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiSpon55860",        "abbr": "SbiSpon55860", "km": 55.860, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Schluderns",          "abbr": "SLU",          "km": 56.146, "tracks": 1, "stop_slow": True,  "stop_fast": False},
    {"name": "SblSpon56860",        "abbr": "SblSpon56860", "km": 56.860, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SblSpon57860",        "abbr": "SblSpon57860", "km": 57.860, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "SbiSpon58860",        "abbr": "SbiSpon58860", "km": 58.860, "tracks": 1, "stop_slow": False, "stop_fast": False},
    {"name": "Mals",                "abbr": "MA",           "km": 59.853, "tracks": 2, "stop_slow": True,  "stop_fast": True},
]

# Lunghezza totale linea
LINE_LENGTH_KM = INFRASTRUCTURE[-1]["km"]  # 59.853 km

# Tempi di percorrenza totale (minuti)
SLOW_TRAVEL_MIN  = 84   # Merano → Malles (e viceversa)
FAST_TRAVEL_MIN  = 58   # Merano → Malles (e viceversa)

# =============================================================================
# STRUTTURE DERIVATE: sezioni di blocco
# =============================================================================
# Una "sezione di blocco" è il tratto tra due elementi consecutivi.
# Ogni sezione ha: km_start, km_end, length_km, tracks_min (minimo binari
# degli estremi → se entrambi hanno 1 binario, la sezione è singola).
# Per semplicità usiamo il numero di binari dell'elemento di inizio.

def build_block_sections():
    """
    Costruisce la lista delle sezioni di blocco come segmenti tra elementi
    consecutivi dell'infrastruttura.
    Restituisce lista di dict: {idx, name_start, name_end, km_start, km_end,
                                 length_km, tracks_start, tracks_end}
    """
    sections = []
    for i in range(len(INFRASTRUCTURE) - 1):
        a = INFRASTRUCTURE[i]
        b = INFRASTRUCTURE[i + 1]
        sections.append({
            "idx":          i,
            "name_start":   a["name"],
            "name_end":     b["name"],
            "abbr_start":   a["abbr"],
            "km_start":     a["km"],
            "km_end":       b["km"],
            "length_km":    b["km"] - a["km"],
            "tracks_start": a["tracks"],
            "tracks_end":   b["tracks"],
        })
    return sections

BLOCK_SECTIONS = build_block_sections()

# =============================================================================
# TRENI: definizioni
# =============================================================================
# Direzione: 'ME→MA' (Merano→Malles) o 'MA→ME' (Malles→Merano)
# Tipo: 'slow' o 'fast'
# Colore plot: 'red', 'green', 'blue'

TRAIN_DEFINITIONS = [
    {"id": "t1", "dir": "ME→MA", "type": "slow", "color": "red",   "label": "Lento R  ME→MA"},
    {"id": "t2", "dir": "ME→MA", "type": "slow", "color": "green", "label": "Lento V  ME→MA"},
    {"id": "t3", "dir": "ME→MA", "type": "fast", "color": "blue",  "label": "Veloce B ME→MA"},
    {"id": "t4", "dir": "MA→ME", "type": "slow", "color": "red",   "label": "Lento R  MA→ME"},
    {"id": "t5", "dir": "MA→ME", "type": "slow", "color": "green", "label": "Lento V  MA→ME"},
    {"id": "t6", "dir": "MA→ME", "type": "fast", "color": "blue",  "label": "Veloce B MA→ME"},
]

# =============================================================================
# CALCOLO GAP MINIMO no-sorpasso
# =============================================================================
# Il treno veloce (ME→MA) parte DOPO il lento. Per non raggiungerlo:
#   lento parte a t_slow, arriva a t_slow + 84
#   veloce parte a t_fast = t_slow + gap, arriva a t_fast + 58
#   Il veloce NON deve mai trovarsi km-wise davanti al lento.
#   Posizione lento al tempo t:  km_slow(t) = (t - t_slow) / 84 * L
#   Posizione veloce al tempo t: km_fast(t) = (t - t_fast) / 58 * L
#   km_fast(t) <= km_slow(t) per ogni t in [t_fast, t_slow+84]
#   → (t-t_fast)/58 <= (t-t_slow)/84
#   → 84*(t-t_fast) <= 58*(t-t_slow)
#   → 84*t - 84*t_fast <= 58*t - 58*t_slow
#   → 26*t <= 84*t_fast - 58*t_slow
#   Il massimo di t nell'intervallo è t_slow+84.
#   → 26*(t_slow+84) <= 84*t_fast - 58*t_slow
#   → 26*t_slow + 26*84 <= 84*t_fast - 58*t_slow
#   → 84*t_fast >= 84*t_slow + 26*84
#   → t_fast >= t_slow + 26
#   Quindi GAP_MIN = 26 minuti.
#
#   Stesso ragionamento per MA→ME (simmetrico).

GAP_MIN_MINUTES = SLOW_TRAVEL_MIN - FAST_TRAVEL_MIN  # 84 - 58 = 26
HEADWAY_MINUTES = 3  # minimum headway between trains following each other in the same direction

# =============================================================================
# GESTIONE SOLUZIONE
# =============================================================================

def train_position_at(departure_abs_min, train_type, direction, time_abs_min):
    """
    Posizione in km del treno al tempo assoluto time_abs_min.
    Traiettoria lineare.
    Restituisce None se il treno non è ancora partito o è già arrivato.
    """
    travel = SLOW_TRAVEL_MIN if train_type == "slow" else FAST_TRAVEL_MIN
    arrival = departure_abs_min + travel

    if time_abs_min < departure_abs_min or time_abs_min > arrival:
        return None

    frac = (time_abs_min - departure_abs_min) / travel
    if direction == "ME→MA":
        return frac * LINE_LENGTH_KM
    else:  # MA→ME
        return (1.0 - frac) * LINE_LENGTH_KM


def generate_all_departures(solution, n_copies=5):
    """
    Dato solution = [t1..t6] (in minuti relativi 0-60 nell'ora di BASE_HOUR_MIN),
    genera tutti i treni per n_copies ore.
    Restituisce lista di dict: {train_def, departure_abs_min}
    """
    trains = []
    for copy_idx in range(n_copies):
        offset = copy_idx * 60
        for i, tdef in enumerate(TRAIN_DEFINITIONS):
            dep = BASE_HOUR_MIN + solution[i] + offset
            trains.append({
                "train_def":       tdef,
                "departure_abs_min": dep,
                "copy_idx":        copy_idx,
            })
    return trains


def find_section_at_km(km):
    """
    Dato un chilometraggio, restituisce l'indice della sezione di blocco
    in cui si trova quel punto.
    """
    for sec in BLOCK_SECTIONS:
        if sec["km_start"] <= km <= sec["km_end"]:
            return sec["idx"]
    # Bordo finale
    return len(BLOCK_SECTIONS) - 1


def find_crossing_km(dep_a, type_a, dir_a, dep_b, type_b, dir_b):
    """
    Calcola il km di incrocio tra due treni in direzioni opposte
    usando traiettorie lineari.
    Restituisce il km di incrocio o None se non si incrociano.

    Treno A (ME→MA): km_A(t) = (t - dep_a)/travel_a * L
    Treno B (MA→ME): km_B(t) = (1 - (t - dep_b)/travel_b) * L

    Incrocio quando km_A(t) = km_B(t):
    (t - dep_a)/travel_a = 1 - (t - dep_b)/travel_b
    (t - dep_a)/travel_a + (t - dep_b)/travel_b = 1
    t*(1/travel_a + 1/travel_b) = 1 + dep_a/travel_a + dep_b/travel_b
    t_cross = (1 + dep_a/travel_a + dep_b/travel_b) / (1/travel_a + 1/travel_b)
    """
    travel_a = SLOW_TRAVEL_MIN if type_a == "slow" else FAST_TRAVEL_MIN
    travel_b = SLOW_TRAVEL_MIN if type_b == "slow" else FAST_TRAVEL_MIN

    # Deve essere dir_a = ME→MA e dir_b = MA→ME (o viceversa)
    # Normalizziamo in modo che a sia ME→MA
    if dir_a == "MA→ME":
        dep_a, dep_b = dep_b, dep_a
        type_a, type_b = type_b, type_a
        travel_a, travel_b = travel_b, travel_a

    denom = (1.0 / travel_a) + (1.0 / travel_b)
    numer = 1.0 + dep_a / travel_a + dep_b / travel_b
    t_cross = numer / denom

    # Verifica che l'incrocio avvenga mentre entrambi sono in viaggio
    arr_a = dep_a + travel_a
    arr_b = dep_b + travel_b
    t_start = max(dep_a, dep_b)
    t_end   = min(arr_a, arr_b)

    if t_cross < t_start or t_cross > t_end:
        return None

    km_cross = (t_cross - dep_a) / travel_a * LINE_LENGTH_KM
    km_cross = max(0.0, min(LINE_LENGTH_KM, km_cross))
    return km_cross


def compute_cost(solution, n_copies=5):
    """
    Calcola il costo totale della soluzione (km di binario da raddoppiare).
    
    Logica:
    1. Genera tutti i treni (n_copies ore).
    2. Per ogni coppia (treno ME→MA, treno MA→ME), trova il km di incrocio.
    3. Individua la sezione di blocco corrispondente.
    4. Se entrambi gli estremi della sezione hanno 2+ binari → costo 0 (stazione doppio binario).
       Se uno dei due estremi è a binario singolo → la sezione va raddoppiata.
    5. Somma le lunghezze delle sezioni da raddoppiare (senza contare due volte).
    
    Restituisce: (costo_totale_km, set di indici sezioni da raddoppiare, lista incroci)
    """
    all_trains = generate_all_departures(solution, n_copies)

    me_ma_trains = [t for t in all_trains if t["train_def"]["dir"] == "ME→MA"]
    ma_me_trains = [t for t in all_trains if t["train_def"]["dir"] == "MA→ME"]

    sections_to_double = set()
    crossings = []

    for a in me_ma_trains:
        for b in ma_me_trains:
            km_cross = find_crossing_km(
                a["departure_abs_min"], a["train_def"]["type"], a["train_def"]["dir"],
                b["departure_abs_min"], b["train_def"]["type"], b["train_def"]["dir"],
            )
            if km_cross is None:
                continue

            sec_idx = find_section_at_km(km_cross)
            sec = BLOCK_SECTIONS[sec_idx]

            crossings.append({
                "km":      km_cross,
                "sec_idx": sec_idx,
                "train_a": a,
                "train_b": b,
            })

            # Costo = 0 se la sezione è interamente in una stazione a 2+ binari
            # (ovvero entrambi gli estremi hanno tracks >= 2)
            if sec["tracks_start"] >= 2 and sec["tracks_end"] >= 2:
                pass  # incrocio gestito gratuitamente
            else:
                sections_to_double.add(sec_idx)

    total_cost = sum(BLOCK_SECTIONS[i]["length_km"] for i in sections_to_double)
    return total_cost, sections_to_double, crossings


# =============================================================================
# VINCOLI: no-sorpasso (applicato nella generazione della soluzione)
# =============================================================================

def enforce_no_overtake(solution):
    """
    Dato un vettore [t1..t6], aggiusta gli orari per garantire il vincolo
    no-sorpasso: il treno veloce deve partire DOPO entrambi i lenti nella
    stessa direzione, con un gap minimo di GAP_MIN minuti.

    Per ME→MA: t1 (lento1), t2 (lento2), t3 (veloce)
    Per MA→ME: t4 (lento1), t5 (lento2), t6 (veloce)

    Regola circolare (periodo 60 min):
    - I due lenti partizionano il cerchio in due archi.
    - Il veloce deve stare nell'arco che inizia GAP_MIN dopo il "secondo lento"
      (quello più avanzato nel senso di marcia temporale) e finisce GAP_MIN
      prima del "primo lento" del ciclo successivo.
    - Se l'arco valido è troppo piccolo (< 2*GAP_MIN), il veloce viene fissato
      al punto medio dell'arco.

    Strategia: i lenti sono fissi, il veloce viene proiettato nella finestra
    valida se necessario.
    """
    s = list(solution)

    def enforce_circular_headway(a, b):
        """Garantisce almeno HEADWAY_MINUTES di gap circolare tra due orari."""
        a = a % 60
        b = b % 60
        gap_ab = (b - a) % 60
        gap_ba = (a - b) % 60
        if gap_ab < gap_ba:
            if gap_ab < HEADWAY_MINUTES:
                b = (a + HEADWAY_MINUTES) % 60
        else:
            if gap_ba < HEADWAY_MINUTES:
                a = (b + HEADWAY_MINUTES) % 60
        return a, b

    for slow1_idx, slow2_idx, fast_idx in [(0, 1, 2), (3, 4, 5)]:
        sl1 = s[slow1_idx] % 60
        sl2 = s[slow2_idx] % 60
        fa  = s[fast_idx]  % 60

        # Garantisce headway minimo tra i due treni lenti nella stessa direzione.
        sl1, sl2 = enforce_circular_headway(sl1, sl2)
        s[slow1_idx] = sl1
        s[slow2_idx] = sl2

        # "Secondo lento" = quello che parte per ultimo nell'ora
        # (il veloce deve partire dopo di lui)
        l_last  = max(sl1, sl2)
        l_first = min(sl1, sl2)

        # Finestra valida per il veloce (circolare, modulo 60):
        #   [l_last + GAP_MIN,  l_first + 60 - GAP_MIN)
        valid_start = (l_last  + GAP_MIN_MINUTES) % 60
        valid_end   = (l_first + 60 - GAP_MIN_MINUTES) % 60

        def in_circular_window(x, start, end):
            """True se x è in [start, end) sulla retta circolare mod 60."""
            x     = x % 60
            start = start % 60
            end   = end   % 60
            if start < end:
                return start <= x < end
            elif start > end:
                return x >= start or x < end
            else:
                return False  # finestra di ampiezza zero

        fa_mod = fa % 60
        if not in_circular_window(fa_mod, valid_start, valid_end):
            fa_mod = valid_start % 60

        s[fast_idx] = fa_mod

    return s


def random_solution():
    """Genera una soluzione casuale rispettando il vincolo no-sorpasso."""
    # Lenti: posizioni random in [0, 60)
    t1 = random.uniform(0, 60)
    t2 = random.uniform(0, 60)
    t4 = random.uniform(0, 60)
    t5 = random.uniform(0, 60)
    # Veloce: posizione random in [0, 60), poi corretta
    t3 = random.uniform(0, 60)
    t6 = random.uniform(0, 60)
    sol = enforce_no_overtake([t1, t2, t3, t4, t5, t6])
    return sol


def perturb_solution(solution):
    """
    Genera una soluzione adiacente perturbando UN SOLO orario di ±SA_PERTURB_MAX_MIN.
    Applica poi il vincolo no-sorpasso.
    """
    s = list(solution)
    idx = random.randint(0, 5)
    delta = random.uniform(-SA_PERTURB_MAX_MIN, SA_PERTURB_MAX_MIN)
    s[idx] = (s[idx] + delta) % 60
    s = enforce_no_overtake(s)
    return s


# =============================================================================
# SIMULATED ANNEALING
# =============================================================================

def simulated_annealing(show_schedules=False, plot_interval=5000):
    """
    Esegue il Simulated Annealing.
    Restituisce (best_solution, best_cost, cost_history).
    """
    current_sol  = random_solution()
    current_cost, _, _ = compute_cost(current_sol)

    best_sol  = list(current_sol)
    best_cost = current_cost

    temp = SA_INITIAL_TEMP
    cost_history = [current_cost]

    print("=== SIMULATED ANNEALING - Merano-Malles ===")
    print(f"Parametri: max_iter={SA_MAX_ITERATIONS}, T0={SA_INITIAL_TEMP}, "
          f"cooling={SA_COOLING_RATE}, perturb_max={SA_PERTURB_MAX_MIN} min")
    print(f"Soluzione iniziale: {[f'{x:.2f}' for x in current_sol]}")
    print(f"Costo iniziale: {current_cost:.4f} km\n")

    for iteration in range(SA_MAX_ITERATIONS):
        if temp < SA_MIN_TEMP:
            print(f"  → Stop anticipato alla iterazione {iteration} (T < T_min)")
            break

        new_sol  = perturb_solution(current_sol)
        new_cost, _, _ = compute_cost(new_sol)
        delta    = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_sol  = new_sol
            current_cost = new_cost

        if current_cost < best_cost:
            best_sol  = list(current_sol)
            best_cost = current_cost

        temp *= SA_COOLING_RATE
        cost_history.append(current_cost)

        if iteration % plot_interval == 0 and iteration > 0:
            plot_results(current_sol, current_cost, cost_history[:iteration+1], filename=f"marey_iter_{iteration}.png", show_plot=False)

        if iteration % 5000 == 0:
            print(f"  Iter {iteration:6d} | T={temp:8.4f} | Cost corrente={current_cost:.4f} | "
                  f"Best={best_cost:.4f}")

    print(f"\n=== RISULTATO FINALE ===")
    print(f"Soluzione ottimale trovata: {[f'{x:.2f}' for x in best_sol]}")
    print(f"  t1 (Lento R  ME→MA): {BASE_HOUR_MIN + best_sol[0]:.1f} min "
          f"→ {int((BASE_HOUR_MIN + best_sol[0])//60):02d}:{int((BASE_HOUR_MIN + best_sol[0])%60):02d}")
    print(f"  t2 (Lento V  ME→MA): {BASE_HOUR_MIN + best_sol[1]:.1f} min "
          f"→ {int((BASE_HOUR_MIN + best_sol[1])//60):02d}:{int((BASE_HOUR_MIN + best_sol[1])%60):02d}")
    print(f"  t3 (Veloce B ME→MA): {BASE_HOUR_MIN + best_sol[2]:.1f} min "
          f"→ {int((BASE_HOUR_MIN + best_sol[2])//60):02d}:{int((BASE_HOUR_MIN + best_sol[2])%60):02d}")
    print(f"  t4 (Lento R  MA→ME): {BASE_HOUR_MIN + best_sol[3]:.1f} min "
          f"→ {int((BASE_HOUR_MIN + best_sol[3])//60):02d}:{int((BASE_HOUR_MIN + best_sol[3])%60):02d}")
    print(f"  t5 (Lento V  MA→ME): {BASE_HOUR_MIN + best_sol[4]:.1f} min "
          f"→ {int((BASE_HOUR_MIN + best_sol[4])//60):02d}:{int((BASE_HOUR_MIN + best_sol[4])%60):02d}")
    print(f"  t6 (Veloce B MA→ME): {BASE_HOUR_MIN + best_sol[5]:.1f} min "
          f"→ {int((BASE_HOUR_MIN + best_sol[5])//60):02d}:{int((BASE_HOUR_MIN + best_sol[5])%60):02d}")
    print(f"Costo totale (km da raddoppiare): {best_cost:.4f} km")

    return best_sol, best_cost, cost_history


# =============================================================================
# VISUALIZZAZIONE
# =============================================================================

def minutes_to_hhmm(m):
    h = int(m) // 60
    mi = int(m) % 60
    return f"{h:02d}:{mi:02d}"


def plot_results(best_sol, best_cost, cost_history, filename="merano_malles_risultato.png", show_plot=True):
    """
    Produce due grafici:
    1. Marey Diagram (finestra 10:00-13:00)
    2. Convergenza del costo SA
    """
    _, sections_to_double, crossings = compute_cost(best_sol, n_copies=5)
    all_trains = generate_all_departures(best_sol, n_copies=5)

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Ottimizzazione raddoppio Merano–Malles  |  Simulated Annealing",
                 fontsize=14, fontweight="bold")

    # ------------------------------------------------------------------
    # Subplot 1: Marey Diagram
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(1, 2, 1)

    # Asse x = km, asse y = tempo (minuti assoluti, 10:00 - 13:00)
    ax1.set_xlim(0, LINE_LENGTH_KM)
    ax1.set_ylim(VIZ_START_MIN, VIZ_END_MIN)
    ax1.invert_yaxis()  # tempo cresce verso il basso (convenzione Marey)
    ax1.set_xlabel("Chilometraggio [km]", fontsize=10)
    ax1.set_ylabel("Ora", fontsize=10)
    ax1.set_title(f"Diagramma di Marey  (10:00–13:00)\nCosto soluzione: {best_cost:.3f} km da raddoppiare",
                  fontsize=11)

    # Griglia leggera
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)

    # Asse Y: etichette orarie
    y_ticks = range(VIZ_START_MIN, VIZ_END_MIN + 1, 10)
    ax1.set_yticks(list(y_ticks))
    ax1.set_yticklabels([minutes_to_hhmm(m) for m in y_ticks], fontsize=7)

    # --- Sezioni di blocco (tacche sull'asse x) ---
    for node in INFRASTRUCTURE:
        km = node["km"]
        is_station = node["stop_slow"] or node["stop_fast"]
        # Linea verticale leggera per ogni nodo
        ax1.axvline(x=km, color="gray", linewidth=0.3, alpha=0.4)

    # Evidenzia sezioni da raddoppiare in rosso (sfondo)
    for sec_idx in sections_to_double:
        sec = BLOCK_SECTIONS[sec_idx]
        ax1.axvspan(sec["km_start"], sec["km_end"],
                    alpha=0.18, color="red", zorder=0)

    # Tacche asse x: solo per le stazioni (con abbreviazione)
    station_nodes = [n for n in INFRASTRUCTURE if n["stop_slow"] or n["stop_fast"]]
    ax1.set_xticks([n["km"] for n in station_nodes])
    ax1.set_xticklabels([n["abbr"] for n in station_nodes],
                        rotation=90, fontsize=7)

    # Linea rossa nella legenda per sezioni da raddoppiare
    double_patch = mpatches.Patch(color="red", alpha=0.35,
                                  label=f"Sezioni da raddoppiare ({best_cost:.2f} km)")

    # --- Treni: disegna solo quelli nella finestra 10:00-13:00 ---
    # Mappa colore + stile per direzione
    color_map  = {"red": "red", "green": "green", "blue": "blue"}
    style_me_ma = {"linestyle": "-",  "linewidth": 1.8}
    style_ma_me = {"linestyle": "--", "linewidth": 1.8}

    # Offset visivo (minuti) per separare treni quasi sovrapposti nel plot
    VISUAL_OFFSET = {"t1": -0.4, "t2": 0.4, "t3": 0.0,
                     "t4": -0.4, "t5": 0.4, "t6": 0.0}

    legend_handles = {}
    for train in all_trains:
        tdef = train["train_def"]
        dep  = train["departure_abs_min"]
        trav = SLOW_TRAVEL_MIN if tdef["type"] == "slow" else FAST_TRAVEL_MIN
        arr  = dep + trav
        voff = VISUAL_OFFSET.get(tdef["id"], 0.0)

        # Clippa alla finestra visuale
        t_start_vis = max(dep, VIZ_START_MIN)
        t_end_vis   = min(arr, VIZ_END_MIN)
        if t_start_vis >= t_end_vis:
            continue

        frac_start = (t_start_vis - dep) / trav
        frac_end   = (t_end_vis   - dep) / trav

        if tdef["dir"] == "ME→MA":
            km_start_vis = frac_start * LINE_LENGTH_KM
            km_end_vis   = frac_end   * LINE_LENGTH_KM
        else:
            km_start_vis = (1 - frac_start) * LINE_LENGTH_KM
            km_end_vis   = (1 - frac_end)   * LINE_LENGTH_KM

        style = style_me_ma if tdef["dir"] == "ME→MA" else style_ma_me
        col   = color_map[tdef["color"]]

        ax1.plot(
            [km_start_vis, km_end_vis],
            [t_start_vis + voff, t_end_vis + voff],
            color=col, **style, alpha=0.90, zorder=3
        )

        leg_key = tdef["id"]
        if leg_key not in legend_handles:
            from matplotlib.lines import Line2D
            linestyle = "-" if tdef["dir"] == "ME→MA" else "--"
            legend_handles[leg_key] = Line2D(
                [0], [0], color=col, linewidth=2,
                linestyle=linestyle, label=tdef["label"]
            )

    # Punti di incrocio (solo nella finestra)
    for cr in crossings:
        dep_a   = cr["train_a"]["departure_abs_min"]
        type_a  = cr["train_a"]["train_def"]["type"]
        trav_a  = SLOW_TRAVEL_MIN if type_a == "slow" else FAST_TRAVEL_MIN
        t_cross_abs = dep_a + (cr["km"] / LINE_LENGTH_KM) * trav_a
        if VIZ_START_MIN <= t_cross_abs <= VIZ_END_MIN:
            ax1.plot(cr["km"], t_cross_abs, "kx", markersize=6, zorder=5)

    handles_list = list(legend_handles.values()) + [double_patch]
    ax1.legend(handles=handles_list, loc="lower right", fontsize=7,
               framealpha=0.85)

    # ------------------------------------------------------------------
    # Subplot 2: Convergenza SA
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(cost_history, color="steelblue", linewidth=0.8, alpha=0.8)
    ax2.set_xlabel("Iterazione", fontsize=10)
    ax2.set_ylabel("Costo (km da raddoppiare)", fontsize=10)
    ax2.set_title("Convergenza Simulated Annealing", fontsize=11)
    ax2.axhline(y=best_cost, color="red", linestyle="--", linewidth=1.2,
                label=f"Best = {best_cost:.3f} km")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    # Stampa sezioni da raddoppiare
    print(f"\nSezioni di blocco da raddoppiare ({len(sections_to_double)} sezioni):")
    for sec_idx in sorted(sections_to_double):
        sec = BLOCK_SECTIONS[sec_idx]
        print(f"  [{sec_idx:2d}] {sec['name_start']:25s} → {sec['name_end']:25s} "
              f"| {sec['km_start']:.3f}–{sec['km_end']:.3f} km "
              f"| lunghezza {sec['length_km']:.3f} km")

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"\nGrafico salvato in: {filename}")
    if show_plot:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    seed = 12 # Cambia questo seed per ottenere risultati diversi
    random.seed(seed)
    np.random.seed(seed)

    best_sol, best_cost, cost_history = simulated_annealing(show_schedules=True)
    plot_results(best_sol, best_cost, cost_history)