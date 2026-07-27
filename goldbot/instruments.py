"""Registro de instrumentos: oro y divisas.

El sistema nacio para XAU/USD, pero la logica de descubrimiento, validacion y
riesgo no tiene nada de especifico del oro: todo esta expresado en unidades de
precio y multiplos de ATR. Lo unico que cambia entre instrumentos son cuatro
constantes -- y precisamente por eso conviene tenerlas en un solo sitio, porque
son las que arruinan una cuenta si estan mal.

Lo que de verdad cambia:

* **Tamano de contrato.** Un lote de oro son 100 onzas; uno de EUR/USD son
  100.000 euros. Con el mismo numero de lotes, el riesgo en dolares difiere en
  tres ordenes de magnitud. Si esta constante esta mal, el dimensionamiento de
  posiciones tambien lo esta, y en la direccion peligrosa.
* **Escala de precio.** El oro cotiza a miles; el euro, a ~1,08. Un spread de
  0,25 es normal en oro y absurdo en divisas.
* **Rango de cordura.** Sirve para que el bot detecte que el broker le esta
  dando el precio de otro instrumento, cosa que pasa cuando el simbolo se
  resuelve mal.
* **Sesion util.** El oro y el euro comparten el solape Londres/Nueva York,
  pero el euro se mueve mucho menos en la sesion asiatica.

Para operar los dos a la vez, lo natural es un proceso por instrumento con su
propio YAML: cada uno mantiene su cache, su base de datos y su campeon. Mezclar
dos instrumentos en un mismo motor evolutivo produciria estrategias promedio que
no funcionan bien en ninguno.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentSpec:
    """Todo lo que el sistema necesita saber de un instrumento."""

    name: str
    description: str
    aliases: tuple[str, ...]

    # --- contrato ---
    contract_size: float          # unidades del activo por lote estandar
    point: float                  # incremento minimo de precio
    pip: float                    # movimiento de referencia del mercado

    # --- rango de cordura ---
    price_min: float
    price_max: float

    # --- costes tipicos (punto de partida; ajustar a los del broker real) ---
    spread_points: float          # en unidades de precio
    slippage_points: float
    commission_per_lot: float     # USD ida y vuelta
    swap_long: float
    swap_short: float

    # --- proveedores de datos alternativos a MT5 ---
    yfinance_symbol: str = ""
    yfinance_fallback: str = ""
    ccxt_symbol: str = ""

    @property
    def spread_pips(self) -> float:
        return self.spread_points / self.pip if self.pip else 0.0

    def is_plausible_price(self, price: float) -> bool:
        """Detecta que el broker esta cotizando otro instrumento."""
        return self.price_min < price < self.price_max

    def round_trip_cost(self) -> float:
        """Coste total de ida y vuelta en unidades de precio."""
        commission_in_price = self.commission_per_lot / self.contract_size
        return self.spread_points + 2 * self.slippage_points + commission_in_price

    def summary(self) -> str:
        return (
            f"{self.name} ({self.description}) | contrato {self.contract_size:,.0f} | "
            f"spread {self.spread_points:g} ({self.spread_pips:.1f} pips) | "
            f"coste ida y vuelta {self.round_trip_cost():g}"
        )


# --------------------------------------------------------------------------- #
# Catalogo
# --------------------------------------------------------------------------- #
XAUUSD = InstrumentSpec(
    name="XAUUSD",
    description="Oro contra dolar",
    # XM publica el oro como GOLD; Vantage como XAUUSD+ en cuentas STP.
    aliases=(
        "XAUUSD", "GOLD", "XAUUSD+", "XAUUSD.a", "XAUUSDm", "XAUUSD_i",
        "XAUUSD.", "XAUUSDx", "XAUUSD#", "GOLDmicro", "GOLD.", "XAUUSDc",
    ),
    contract_size=100.0,          # onzas troy por lote
    point=0.01,
    pip=0.1,                      # convenio habitual en oro
    price_min=100.0,
    price_max=20_000.0,
    spread_points=0.25,           # USD por onza
    slippage_points=0.10,
    commission_per_lot=7.0,
    swap_long=-0.5,
    swap_short=0.2,
    yfinance_symbol="GC=F",
    yfinance_fallback="XAUUSD=X",
    ccxt_symbol="PAXG/USDT",
)

EURUSD = InstrumentSpec(
    name="EURUSD",
    description="Euro contra dolar",
    aliases=(
        "EURUSD", "EURUSD+", "EURUSD.a", "EURUSDm", "EURUSD_i",
        "EURUSD.", "EURUSDx", "EURUSD#", "EURUSDc", "EURUSDmicro",
    ),
    contract_size=100_000.0,      # euros por lote estandar
    point=0.00001,                # cotizacion a 5 decimales
    pip=0.0001,
    price_min=0.5,
    price_max=2.0,
    # ~1 pip de spread es lo normal en cuentas estandar de XM/Vantage.
    spread_points=0.00010,
    slippage_points=0.00002,
    commission_per_lot=0.0,       # las cuentas estandar no cobran comision
    swap_long=-0.8,
    swap_short=0.1,
    yfinance_symbol="EURUSD=X",
    yfinance_fallback="EUR=X",
    ccxt_symbol="EUR/USDT",
)

REGISTRY: dict[str, InstrumentSpec] = {
    "XAUUSD": XAUUSD,
    "GOLD": XAUUSD,
    "ORO": XAUUSD,
    "EURUSD": EURUSD,
    "EUR": EURUSD,
}


def get_instrument(name: str) -> InstrumentSpec:
    """Busca un instrumento por nombre o alias.

    La busqueda es tolerante para que el usuario pueda escribir ``eurusd``,
    ``EUR/USD`` o ``EURUSD+`` sin que el bot se niegue a arrancar.
    """
    if not name:
        return XAUUSD

    key = name.upper().replace("/", "").replace("-", "").strip()
    if key in REGISTRY:
        return REGISTRY[key]

    # Coincidencia por alias del broker (p.ej. "XAUUSD+" -> XAUUSD).
    for spec in {id(s): s for s in REGISTRY.values()}.values():
        if key in {alias.upper() for alias in spec.aliases}:
            return spec

    # Coincidencia parcial: cubre sufijos de cuenta no catalogados.
    for spec in {id(s): s for s in REGISTRY.values()}.values():
        if key.startswith(spec.name):
            return spec

    known = sorted({s.name for s in REGISTRY.values()})
    raise ValueError(f"Instrumento desconocido: {name!r}. Conocidos: {known}")


def detect_from_price(price: float) -> InstrumentSpec | None:
    """Deduce el instrumento a partir de su cotizacion.

    Ultimo recurso de diagnostico: si el simbolo se resolvio mal, comparar el
    precio con los rangos conocidos suele revelar que se cogio otro mercado.
    """
    for spec in {id(s): s for s in REGISTRY.values()}.values():
        if spec.is_plausible_price(price):
            return spec
    return None


def all_instruments() -> list[InstrumentSpec]:
    """Instrumentos unicos del registro (sin repetir por alias)."""
    return list({id(s): s for s in REGISTRY.values()}.values())
