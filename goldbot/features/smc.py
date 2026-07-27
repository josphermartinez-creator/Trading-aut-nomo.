"""Smart Money Concepts: estructura, liquidez, order blocks y desequilibrios.

Traduccion de los conceptos de SMC a series numericas que el algoritmo genetico
puede combinar con el resto de features.

**El punto delicado de todo este modulo es la causalidad de los swings.** Un
maximo estructural no se "conoce" en la vela en la que ocurre: hacen falta N
velas posteriores que no lo superen para confirmarlo. Casi todas las
implementaciones de SMC que circulan usan un ``rolling(center=True)`` y marcan
el swing en su propia vela, lo que mete N barras de informacion futura en cada
señal. Sobre eso, el motor evolutivo construye estrategias espectaculares e
irreproducibles.

Aqui la confirmacion se desplaza siempre a la barra ``t`` mirando hacia atras:
en ``t`` preguntamos si la barra ``t - right`` fue un extremo dentro de una
ventana que **termina en t**. Todos los datos usados son pasados.

Convenios del modulo:

* Las funciones devuelven series alineadas con el indice de entrada.
* Los precios se normalizan por el cierre antes de exponerse como feature, para
  que sigan siendo comparables con el oro a 1.800 o a 3.000 USD.
* Las zonas (order blocks, FVG) se siguen "la mas reciente sin mitigar", que es
  como se opera en la practica y ademas mantiene el coste en O(n).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Cuantas velas hacia atras se busca la vela de origen de un order block.
_OB_LOOKBACK = 20


# --------------------------------------------------------------------------- #
# Swings confirmados
# --------------------------------------------------------------------------- #
def swing_points(
    high: pd.Series, low: pd.Series, left: int = 3, right: int = 3
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Swings **confirmados**, sin mirar al futuro.

    En la barra ``t`` se comprueba si la barra ``t - right`` fue el maximo (o
    minimo) de una ventana que termina en ``t``. Como la ventana no se extiende
    mas alla de ``t``, la comprobacion solo usa informacion ya disponible: el
    swing existia desde antes, pero el mercado no lo habia confirmado hasta
    ahora, que es exactamente como lo vive un operador real.

    Returns
    -------
    (nivel_ultimo_maximo, nivel_ultimo_minimo, flag_confirma_maximo, flag_confirma_minimo)
        Los dos primeros vienen propagados hacia adelante (``ffill``): en cada
        barra indican el ultimo swing conocido hasta ese momento.
    """
    window = left + right + 1

    rolling_max = high.rolling(window, min_periods=window).max()
    rolling_min = low.rolling(window, min_periods=window).min()

    # high.shift(right) en la barra t es el maximo de la barra t-right.
    candidate_high = high.shift(right)
    candidate_low = low.shift(right)

    confirmed_high = (candidate_high >= rolling_max).fillna(False)
    confirmed_low = (candidate_low <= rolling_min).fillna(False)

    last_high = candidate_high.where(confirmed_high).ffill()
    last_low = candidate_low.where(confirmed_low).ffill()

    return last_high, last_low, confirmed_high, confirmed_low


def previous_swing_levels(
    last_high: pd.Series, last_low: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Penultimo swing de cada lado.

    Necesario para detectar maximos/minimos iguales (bolsas de liquidez) y para
    comparar si la estructura hace maximos crecientes o decrecientes.
    """
    prev_high = last_high.where(last_high != last_high.shift(1)).ffill().shift(1).ffill()
    prev_low = last_low.where(last_low != last_low.shift(1)).ffill().shift(1).ffill()
    return prev_high, prev_low


# --------------------------------------------------------------------------- #
# Estructura de mercado: BOS y CHoCH
# --------------------------------------------------------------------------- #
def market_structure(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    left: int = 3,
    right: int = 3,
) -> dict[str, pd.Series]:
    """Estructura de mercado y order blocks asociados, en una sola pasada.

    * **BOS** (Break of Structure): el precio rompe el ultimo swing en la misma
      direccion que la estructura vigente. Es continuacion.
    * **CHoCH** (Change of Character): rompe en direccion contraria a la
      estructura vigente. Es el primer aviso de giro.

    Un nivel roto no vuelve a disparar señal hasta que se forma un swing nuevo:
    sin esa condicion, una tendencia sostenida generaria un BOS en cada vela.

    El **order block** se identifica como la ultima vela de color opuesto antes
    del impulso que rompio la estructura: es la zona donde se supone que el
    dinero institucional dejo ordenes sin ejecutar.
    """
    last_high, last_low, _, _ = swing_points(high, low, left, right)

    sh = last_high.to_numpy(dtype="float64")
    sl = last_low.to_numpy(dtype="float64")
    c = close.to_numpy(dtype="float64")
    o = open_.to_numpy(dtype="float64")
    h = high.to_numpy(dtype="float64")
    lo = low.to_numpy(dtype="float64")
    n = len(c)

    structure = np.zeros(n)
    bos = np.zeros(n)
    choch = np.zeros(n)
    ob_bull_top = np.full(n, np.nan)
    ob_bull_bottom = np.full(n, np.nan)
    ob_bear_top = np.full(n, np.nan)
    ob_bear_bottom = np.full(n, np.nan)

    direction = 0.0
    broken_high = np.nan
    broken_low = np.nan
    bull_top = bull_bottom = bear_top = bear_bottom = np.nan

    for i in range(1, n):
        level_high, level_low = sh[i], sl[i]

        # --- ruptura alcista --- #
        if np.isfinite(level_high) and c[i] > level_high and level_high != broken_high:
            if direction < 0:
                choch[i] = 1.0
            else:
                bos[i] = 1.0
            direction = 1.0
            broken_high = level_high

            # Order block alcista: ultima vela bajista antes del impulso.
            start = max(0, i - _OB_LOOKBACK)
            j = i
            while j > start and c[j] >= o[j]:
                j -= 1
            bull_top, bull_bottom = h[j], lo[j]

        # --- ruptura bajista --- #
        elif np.isfinite(level_low) and c[i] < level_low and level_low != broken_low:
            if direction > 0:
                choch[i] = 1.0
            else:
                bos[i] = 1.0
            direction = -1.0
            broken_low = level_low

            start = max(0, i - _OB_LOOKBACK)
            j = i
            while j > start and c[j] <= o[j]:
                j -= 1
            bear_top, bear_bottom = h[j], lo[j]

        # Un order block se considera mitigado cuando el precio lo atraviesa
        # por completo: a partir de ahi deja de ser una zona de interes.
        if np.isfinite(bull_bottom) and c[i] < bull_bottom:
            bull_top = bull_bottom = np.nan
        if np.isfinite(bear_top) and c[i] > bear_top:
            bear_top = bear_bottom = np.nan

        structure[i] = direction
        ob_bull_top[i], ob_bull_bottom[i] = bull_top, bull_bottom
        ob_bear_top[i], ob_bear_bottom[i] = bear_top, bear_bottom

    index = close.index
    return {
        "structure": pd.Series(structure, index=index),
        "bos": pd.Series(bos, index=index),
        "choch": pd.Series(choch, index=index),
        "ob_bull_top": pd.Series(ob_bull_top, index=index),
        "ob_bull_bottom": pd.Series(ob_bull_bottom, index=index),
        "ob_bear_top": pd.Series(ob_bear_top, index=index),
        "ob_bear_bottom": pd.Series(ob_bear_bottom, index=index),
        "last_swing_high": last_high,
        "last_swing_low": last_low,
    }


# --------------------------------------------------------------------------- #
# Toma de liquidez (barrido de stops)
# --------------------------------------------------------------------------- #
def liquidity_sweep(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    last_swing_high: pd.Series,
    last_swing_low: pd.Series,
    min_penetration: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Barridos de liquidez: la mecha perfora un swing previo y el cierre vuelve.

    Es el "stop hunt" clasico. Por encima de un maximo relevante se acumulan
    stops de cortos y ordenes de compra; el precio va a buscarlos, los ejecuta y
    se da la vuelta. Que el **cierre** regrese al lado correcto es lo que
    distingue un barrido de una ruptura genuina, y por eso la condicion se
    evalua sobre el cierre de la vela ya formada.

    Returns
    -------
    (barrido_alcista, barrido_bajista)
        ``barrido_alcista`` marca liquidez tomada **por debajo** (mechas bajo un
        minimo previo con cierre por encima): sesgo de giro al alza.
        ``barrido_bajista`` es el simetrico por arriba.
    """
    swept_below = (
        (low < last_swing_low * (1 - min_penetration))
        & (close > last_swing_low)
        & last_swing_low.notna()
    )
    swept_above = (
        (high > last_swing_high * (1 + min_penetration))
        & (close < last_swing_high)
        & last_swing_high.notna()
    )
    return swept_below.fillna(False).astype("float64"), swept_above.fillna(False).astype("float64")


def equal_levels(
    last_swing_high: pd.Series,
    last_swing_low: pd.Series,
    atr: pd.Series,
    tolerance_atr: float = 0.25,
) -> tuple[pd.Series, pd.Series]:
    """Maximos y minimos iguales: bolsas de liquidez en reposo.

    Dos swings al mismo nivel concentran stops. Son los objetivos naturales de
    un barrido, asi que saber que existen (y a que distancia) es informacion
    operativa util.
    """
    prev_high, prev_low = previous_swing_levels(last_swing_high, last_swing_low)
    tolerance = (atr * tolerance_atr).replace(0, np.nan)

    equal_highs = ((last_swing_high - prev_high).abs() <= tolerance).fillna(False)
    equal_lows = ((last_swing_low - prev_low).abs() <= tolerance).fillna(False)

    return equal_highs.astype("float64"), equal_lows.astype("float64")


# --------------------------------------------------------------------------- #
# Desequilibrios (Fair Value Gaps)
# --------------------------------------------------------------------------- #
def fair_value_gaps(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> dict[str, pd.Series]:
    """Fair Value Gaps: huecos de tres velas que el precio tiende a rellenar.

    Un FVG alcista existe cuando el minimo de la vela actual queda por encima
    del maximo de la vela ``t-2``: entre ambos no hubo negociacion. El patron se
    detecta en la tercera vela, de modo que es causal por construccion.

    Se sigue el hueco mas reciente sin mitigar de cada lado. Cuando el precio
    entra en la zona, el hueco se considera rellenado y se descarta.
    """
    h = high.to_numpy(dtype="float64")
    lo = low.to_numpy(dtype="float64")
    c = close.to_numpy(dtype="float64")
    n = len(c)

    bull_top = np.full(n, np.nan)
    bull_bottom = np.full(n, np.nan)
    bear_top = np.full(n, np.nan)
    bear_bottom = np.full(n, np.nan)

    active_bull_top = active_bull_bottom = np.nan
    active_bear_top = active_bear_bottom = np.nan

    for i in range(2, n):
        # FVG alcista: hueco entre el maximo de i-2 y el minimo de i.
        if lo[i] > h[i - 2]:
            active_bull_top, active_bull_bottom = lo[i], h[i - 2]
        # FVG bajista: hueco entre el minimo de i-2 y el maximo de i.
        if h[i] < lo[i - 2]:
            active_bear_top, active_bear_bottom = lo[i - 2], h[i]

        # Mitigacion: el precio vuelve a entrar en el hueco.
        if np.isfinite(active_bull_bottom) and c[i] < active_bull_bottom:
            active_bull_top = active_bull_bottom = np.nan
        if np.isfinite(active_bear_top) and c[i] > active_bear_top:
            active_bear_top = active_bear_bottom = np.nan

        bull_top[i], bull_bottom[i] = active_bull_top, active_bull_bottom
        bear_top[i], bear_bottom[i] = active_bear_top, active_bear_bottom

    index = close.index
    return {
        "bull_top": pd.Series(bull_top, index=index),
        "bull_bottom": pd.Series(bull_bottom, index=index),
        "bear_top": pd.Series(bear_top, index=index),
        "bear_bottom": pd.Series(bear_bottom, index=index),
    }


# --------------------------------------------------------------------------- #
# Rango operativo: premium / discount
# --------------------------------------------------------------------------- #
def premium_discount(
    close: pd.Series, last_swing_high: pd.Series, last_swing_low: pd.Series
) -> pd.Series:
    """Posicion dentro del rango operativo actual, en [0, 1].

    0 = minimo del rango (descuento profundo), 1 = maximo (premium). El punto
    medio es el equilibrio. La logica SMC dice comprar en descuento y vender en
    premium; expuesto como feature continua, el genetico puede decidir si esa
    regla aporta algo o no.
    """
    span = (last_swing_high - last_swing_low).replace(0, np.nan)
    position = (close - last_swing_low) / span
    return position.clip(lower=-0.5, upper=1.5)


def displacement(
    open_: pd.Series, close: pd.Series, atr: pd.Series
) -> pd.Series:
    """Tamaño del cuerpo relativo al ATR: mide intencion, no solo movimiento.

    Un desplazamiento fuerte tras un barrido de liquidez es la firma tipica de
    una entrada institucional; una vela grande aislada, en cambio, suele ser
    ruido de noticia.
    """
    body = (close - open_).abs()
    return (body / atr.replace(0, np.nan)).clip(upper=10.0)


# --------------------------------------------------------------------------- #
# Agregador
# --------------------------------------------------------------------------- #
def compute_smc(
    open_: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    left: int = 3,
    right: int = 3,
) -> dict[str, pd.Series]:
    """Calcula el bloque SMC completo, ya normalizado para usarse como feature.

    Las distancias se dividen por el precio para que sean escala-invariantes y
    las zonas se exponen como binarias "el precio esta dentro", que es la
    pregunta operativa real.
    """
    structure = market_structure(open_, high, low, close, left, right)
    last_high = structure["last_swing_high"]
    last_low = structure["last_swing_low"]

    sweep_bull, sweep_bear = liquidity_sweep(high, low, close, last_high, last_low)
    equal_highs, equal_lows = equal_levels(last_high, last_low, atr)
    fvg = fair_value_gaps(high, low, close)

    out: dict[str, pd.Series] = {}

    # --- estructura --- #
    out["smc_structure"] = structure["structure"]
    # Las rupturas se propagan unas velas: una señal de una sola barra es casi
    # imposible de capturar y el genetico no podria construir nada con ella.
    out["smc_bos_recent"] = structure["bos"].rolling(6, min_periods=1).max()
    out["smc_choch_recent"] = structure["choch"].rolling(6, min_periods=1).max()

    # --- liquidez --- #
    out["smc_sweep_bull"] = sweep_bull.rolling(3, min_periods=1).max()
    out["smc_sweep_bear"] = sweep_bear.rolling(3, min_periods=1).max()
    out["smc_equal_highs"] = equal_highs
    out["smc_equal_lows"] = equal_lows

    # Distancia a la liquidez en reposo: hacia donde "tira" el precio.
    out["smc_dist_swing_high"] = (last_high - close) / close
    out["smc_dist_swing_low"] = (close - last_low) / close

    # --- zonas --- #
    out["smc_premium"] = premium_discount(close, last_high, last_low)
    out["smc_displacement"] = displacement(open_, close, atr)

    out["smc_in_bull_ob"] = _inside_zone(
        close, structure["ob_bull_bottom"], structure["ob_bull_top"]
    )
    out["smc_in_bear_ob"] = _inside_zone(
        close, structure["ob_bear_bottom"], structure["ob_bear_top"]
    )
    out["smc_in_bull_fvg"] = _inside_zone(close, fvg["bull_bottom"], fvg["bull_top"])
    out["smc_in_bear_fvg"] = _inside_zone(close, fvg["bear_bottom"], fvg["bear_top"])

    return out


def _inside_zone(price: pd.Series, bottom: pd.Series, top: pd.Series) -> pd.Series:
    """1.0 si el precio esta dentro de la zona activa, 0.0 en caso contrario."""
    inside = (price >= bottom) & (price <= top) & bottom.notna() & top.notna()
    return inside.fillna(False).astype("float64")
