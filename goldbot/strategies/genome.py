"""Genoma de estrategia: la representacion que el algoritmo genetico evoluciona.

Una estrategia se codifica como un arbol poco profundo y legible:

    FILTROS (regimen)  AND  ( REGLAS_LARGO )   -> senal larga
    FILTROS (regimen)  AND  ( REGLAS_CORTO )   -> senal corta
    + REGLAS DE SALIDA (stop/objetivo/trailing/tiempo en ATR)

Se eligio una representacion plana y no programacion genetica con arboles
arbitrarios por tres motivos: (1) el espacio de busqueda queda acotado y no
degenera en formulas ilegibles, (2) toda estrategia resultante se puede leer en
castellano y auditar, y (3) el sobreajuste es mucho menor con pocos grados de
libertad.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from goldbot.backtest.engine import ExitRules
from goldbot.config import EvolutionConfig
from goldbot.features.engineering import FeatureCatalog
from goldbot.features.trend import TREND_COLUMN, apply_trend_veto
from goldbot.strategies.base import Strategy
from goldbot.utils.logging import get_logger

logger = get_logger(__name__)

OPERATORS = ("gt", "lt", "cross_above", "cross_below")
LOGIC_MODES = ("and", "or", "majority")


# --------------------------------------------------------------------------- #
# Condicion
# --------------------------------------------------------------------------- #
@dataclass
class Condition:
    """Una comparacion elemental sobre una feature."""

    feature: str
    operator: str = "gt"
    target_type: str = "const"        # 'const' o 'feature'
    threshold: float = 0.0
    target_feature: str | None = None

    def evaluate(self, features: pd.DataFrame) -> pd.Series:
        """Mascara booleana por barra. Devuelve todo False si falta la feature."""
        if self.feature not in features.columns:
            return pd.Series(False, index=features.index)

        left = features[self.feature]
        if self.target_type == "feature":
            if not self.target_feature or self.target_feature not in features.columns:
                return pd.Series(False, index=features.index)
            right = features[self.target_feature]
        else:
            right = pd.Series(self.threshold, index=features.index)

        if self.operator == "gt":
            out = left > right
        elif self.operator == "lt":
            out = left < right
        elif self.operator == "cross_above":
            # Cruce: requiere estar por debajo en la barra previa. shift(1) es
            # lo que mantiene la condicion causal.
            out = (left > right) & (left.shift(1) <= right.shift(1))
        elif self.operator == "cross_below":
            out = (left < right) & (left.shift(1) >= right.shift(1))
        else:
            raise ValueError(f"Operador desconocido: {self.operator}")

        return out.fillna(False)

    def describe(self) -> str:
        """Version legible, para logs y auditoria."""
        symbols = {"gt": ">", "lt": "<", "cross_above": "cruza arriba", "cross_below": "cruza abajo"}
        right = self.target_feature if self.target_type == "feature" else f"{self.threshold:.6g}"
        return f"{self.feature} {symbols[self.operator]} {right}"

    def features_used(self) -> set[str]:
        used = {self.feature}
        if self.target_type == "feature" and self.target_feature:
            used.add(self.target_feature)
        return used


# --------------------------------------------------------------------------- #
# Conjunto de reglas
# --------------------------------------------------------------------------- #
@dataclass
class RuleSet:
    """Conjunto de condiciones combinadas con una logica."""

    conditions: list[Condition] = field(default_factory=list)
    logic: str = "and"

    def evaluate(self, features: pd.DataFrame) -> pd.Series:
        if not self.conditions:
            # Un conjunto vacio no filtra nada (neutro para el AND).
            return pd.Series(True, index=features.index)

        masks = [c.evaluate(features) for c in self.conditions]
        stacked = pd.concat(masks, axis=1)

        if self.logic == "and":
            return stacked.all(axis=1)
        if self.logic == "or":
            return stacked.any(axis=1)
        if self.logic == "majority":
            # Mayoria estricta: mas de la mitad de las condiciones activas.
            return stacked.sum(axis=1) > (len(masks) / 2)
        raise ValueError(f"Logica desconocida: {self.logic}")

    def describe(self) -> str:
        if not self.conditions:
            return "(sin condiciones)"
        joiner = {"and": " Y ", "or": " O ", "majority": " / "}[self.logic]
        body = joiner.join(c.describe() for c in self.conditions)
        return f"MAYORIA({body})" if self.logic == "majority" else body

    def features_used(self) -> set[str]:
        used: set[str] = set()
        for c in self.conditions:
            used |= c.features_used()
        return used

    def __len__(self) -> int:
        return len(self.conditions)


# --------------------------------------------------------------------------- #
# Genoma
# --------------------------------------------------------------------------- #
@dataclass
class StrategyGenome(Strategy):
    """Estrategia completa codificada como genoma."""

    long_rules: RuleSet = field(default_factory=RuleSet)
    short_rules: RuleSet = field(default_factory=RuleSet)
    regime_filter: RuleSet = field(default_factory=RuleSet)
    exit_rules: ExitRules = field(default_factory=ExitRules)
    allow_long: bool = True
    allow_short: bool = True
    genome_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    generation: int = 0
    parents: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:  # type: ignore[override]
        return f"genome-{self.genome_id}"

    # -- senales --------------------------------------------------------- #
    def generate_signals(self, features: pd.DataFrame, ohlcv: pd.DataFrame | None = None) -> pd.Series:
        """+1/-1/0 por barra."""
        if features.empty:
            return pd.Series(dtype="float64", index=features.index)

        regime_ok = self.regime_filter.evaluate(features)

        long_mask = (
            (self.long_rules.evaluate(features) & regime_ok)
            if self.allow_long
            else pd.Series(False, index=features.index)
        )
        short_mask = (
            (self.short_rules.evaluate(features) & regime_ok)
            if self.allow_short
            else pd.Series(False, index=features.index)
        )

        signals = pd.Series(0.0, index=features.index)
        signals[long_mask] = 1.0
        signals[short_mask] = -1.0
        # Senal contradictoria (larga y corta a la vez) = sin senal. Es mas
        # honesto que inventar un desempate arbitrario.
        signals[long_mask & short_mask] = 0.0

        # Veto de tendencia. Se aplica al final y desde FUERA del genoma: la
        # columna reservada no aparece en el catalogo, asi que la evolucion no
        # puede construir condiciones sobre ella ni eliminarla. Si estuviera
        # dentro del arbol genetico, el optimizador acabaria descartandola en
        # cuanto encontrase un tramo del historico donde ir a contracorriente
        # rentase mas, y siempre existe ese tramo.
        if TREND_COLUMN in features.columns:
            signals = apply_trend_veto(signals, features[TREND_COLUMN])

        return signals

    # -- introspeccion --------------------------------------------------- #
    def required_features(self) -> list[str]:
        used = self.long_rules.features_used() | self.short_rules.features_used()
        used |= self.regime_filter.features_used()
        return sorted(used)

    def complexity(self) -> int:
        """Numero total de condiciones. Se penaliza en la funcion de fitness."""
        return len(self.long_rules) + len(self.short_rules) + len(self.regime_filter)

    def fingerprint(self) -> str:
        """Hash estructural: identifica genomas equivalentes con distinto id.

        Los umbrales se redondean para que dos estrategias que difieren en el
        sexto decimal no cuenten como diversidad genetica real.
        """
        def _rules(rs: RuleSet) -> list:
            return sorted(
                [
                    c.feature,
                    c.operator,
                    c.target_type,
                    c.target_feature or "",
                    round(float(c.threshold), 4),
                ]
                .__str__()
                for c in rs.conditions
            )

        payload = {
            "long": [_rules(self.long_rules), self.long_rules.logic],
            "short": [_rules(self.short_rules), self.short_rules.logic],
            "regime": [_rules(self.regime_filter), self.regime_filter.logic],
            "exit": [
                round(self.exit_rules.stop_atr, 2),
                round(self.exit_rules.target_atr, 2),
                round(self.exit_rules.trail_atr, 2),
                round(self.exit_rules.breakeven_atr, 2),
                self.exit_rules.max_bars,
                self.exit_rules.exit_on_reverse,
            ],
            "sides": [self.allow_long, self.allow_short],
        }
        raw = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def describe(self) -> str:
        """Explicacion en castellano de que hace la estrategia."""
        lines = [f"Estrategia {self.genome_id} (gen {self.generation})"]
        if self.regime_filter.conditions:
            lines.append(f"  FILTRO DE REGIMEN: {self.regime_filter.describe()}")
        if self.allow_long:
            lines.append(f"  LARGO SI: {self.long_rules.describe()}")
        if self.allow_short:
            lines.append(f"  CORTO SI: {self.short_rules.describe()}")
        e = self.exit_rules
        exit_bits = [f"stop {e.stop_atr:.2f}xATR", f"objetivo {e.target_atr:.2f}xATR"]
        if e.trail_atr > 0:
            exit_bits.append(f"trailing {e.trail_atr:.2f}xATR")
        if e.breakeven_atr > 0:
            exit_bits.append(f"BE tras {e.breakeven_atr:.2f}xATR")
        exit_bits.append(f"max {e.max_bars} barras")
        if e.exit_on_reverse:
            exit_bits.append("salida por senal contraria")
        lines.append(f"  SALIDA: {', '.join(exit_bits)}")
        return "\n".join(lines)

    # -- serializacion --------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "generation": self.generation,
            "parents": self.parents,
            "allow_long": self.allow_long,
            "allow_short": self.allow_short,
            "long_rules": {"logic": self.long_rules.logic,
                           "conditions": [asdict(c) for c in self.long_rules.conditions]},
            "short_rules": {"logic": self.short_rules.logic,
                            "conditions": [asdict(c) for c in self.short_rules.conditions]},
            "regime_filter": {"logic": self.regime_filter.logic,
                              "conditions": [asdict(c) for c in self.regime_filter.conditions]},
            "exit_rules": asdict(self.exit_rules),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrategyGenome:
        def _ruleset(raw: dict | None) -> RuleSet:
            if not raw:
                return RuleSet()
            return RuleSet(
                conditions=[Condition(**c) for c in raw.get("conditions", [])],
                logic=raw.get("logic", "and"),
            )

        return cls(
            long_rules=_ruleset(data.get("long_rules")),
            short_rules=_ruleset(data.get("short_rules")),
            regime_filter=_ruleset(data.get("regime_filter")),
            exit_rules=ExitRules(**data.get("exit_rules", {})).validate(),
            allow_long=data.get("allow_long", True),
            allow_short=data.get("allow_short", True),
            genome_id=data.get("genome_id", uuid.uuid4().hex[:12]),
            generation=data.get("generation", 0),
            parents=data.get("parents", []),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> StrategyGenome:
        return cls.from_dict(json.loads(raw))

    def clone(self) -> StrategyGenome:
        """Copia profunda con identidad nueva."""
        copy = StrategyGenome.from_dict(self.to_dict())
        copy.genome_id = uuid.uuid4().hex[:12]
        copy.parents = [self.genome_id]
        return copy


# --------------------------------------------------------------------------- #
# Generacion aleatoria
# --------------------------------------------------------------------------- #
def random_condition(catalog: FeatureCatalog, rng: np.random.Generator) -> Condition:
    """Condicion aleatoria pero semanticamente valida.

    El catalogo garantiza que no se compare un RSI con un precio: los umbrales
    se muestrean del rango util de cada feature y las comparaciones entre
    features solo se permiten dentro del mismo tipo.
    """
    specs = list(catalog)
    if not specs:
        raise ValueError("Catalogo de features vacio")

    spec = specs[rng.integers(len(specs))]

    # Las binarias solo admiten "es verdadera / es falsa".
    if spec.kind == "binary":
        return Condition(
            feature=spec.name,
            operator="gt" if rng.random() < 0.5 else "lt",
            target_type="const",
            threshold=0.5,
        )

    # Un 20% de las veces comparamos dos features del mismo tipo.
    comparables = catalog.comparable_with(spec.name)
    if comparables and rng.random() < 0.20:
        return Condition(
            feature=spec.name,
            operator=OPERATORS[rng.integers(len(OPERATORS))],
            target_type="feature",
            target_feature=comparables[rng.integers(len(comparables))],
        )

    # Los cruces son mas informativos pero mucho mas escasos: 25%.
    operator = (
        ("cross_above" if rng.random() < 0.5 else "cross_below")
        if rng.random() < 0.25
        else ("gt" if rng.random() < 0.5 else "lt")
    )
    return Condition(
        feature=spec.name,
        operator=operator,
        target_type="const",
        threshold=spec.sample_threshold(rng),
    )


def random_ruleset(
    catalog: FeatureCatalog, rng: np.random.Generator, max_conditions: int = 4, min_conditions: int = 1
) -> RuleSet:
    n = int(rng.integers(min_conditions, max_conditions + 1))
    conditions = [random_condition(catalog, rng) for _ in range(n)]

    # Con una sola condicion la logica es irrelevante; mantenemos 'and'.
    if n == 1:
        logic = "and"
    else:
        logic = str(rng.choice(LOGIC_MODES, p=[0.6, 0.25, 0.15]))
    return RuleSet(conditions=conditions, logic=logic)


def random_exit_rules(rng: np.random.Generator) -> ExitRules:
    """Reglas de salida aleatorias, con ratios riesgo/beneficio razonables."""
    stop = float(rng.uniform(0.8, 3.5))
    # El objetivo se define como multiplo del stop para que R:R sea coherente.
    reward_ratio = float(rng.uniform(0.8, 3.0))
    trail = float(rng.uniform(0.5, 3.0)) if rng.random() < 0.4 else 0.0
    breakeven = float(rng.uniform(0.5, 2.0)) if rng.random() < 0.3 else 0.0
    return ExitRules(
        stop_atr=stop,
        target_atr=stop * reward_ratio,
        trail_atr=trail,
        breakeven_atr=breakeven,
        max_bars=int(rng.integers(12, 288)),
        exit_on_reverse=bool(rng.random() < 0.5),
    ).validate()


def random_genome(
    catalog: FeatureCatalog,
    rng: np.random.Generator,
    max_conditions: int = 4,
    generation: int = 0,
) -> StrategyGenome:
    """Crea una estrategia completamente nueva."""
    # Mayoritariamente bidireccional: el oro no tiene sesgo alcista estructural
    # a escala intradia, y restringir el lado desperdicia oportunidades.
    roll = rng.random()
    allow_long, allow_short = (True, True) if roll < 0.7 else ((True, False) if roll < 0.85 else (False, True))

    long_rules = random_ruleset(catalog, rng, max_conditions) if allow_long else RuleSet()
    short_rules = random_ruleset(catalog, rng, max_conditions) if allow_short else RuleSet()

    # El filtro de regimen es opcional: no todas las estrategias lo necesitan.
    regime = (
        random_ruleset(catalog, rng, max_conditions=2)
        if rng.random() < 0.5
        else RuleSet()
    )

    return StrategyGenome(
        long_rules=long_rules,
        short_rules=short_rules,
        regime_filter=regime,
        exit_rules=random_exit_rules(rng),
        allow_long=allow_long,
        allow_short=allow_short,
        generation=generation,
    )


# --------------------------------------------------------------------------- #
# Operadores geneticos
# --------------------------------------------------------------------------- #
def mutate(
    genome: StrategyGenome,
    catalog: FeatureCatalog,
    rng: np.random.Generator,
    config: EvolutionConfig,
    strength: float = 1.0,
) -> StrategyGenome:
    """Devuelve una copia mutada.

    ``strength`` en [0, 1] modula cuanto se mueve: la evolucion arranca con
    mutaciones agresivas (exploracion) y las va suavizando (explotacion).
    """
    child = genome.clone()
    child.generation = genome.generation + 1

    # Cada tipo de mutacion se aplica de forma independiente.
    if rng.random() < 0.35:
        _mutate_ruleset(child.long_rules, catalog, rng, config, strength)
    if rng.random() < 0.35:
        _mutate_ruleset(child.short_rules, catalog, rng, config, strength)
    if rng.random() < 0.20:
        _mutate_ruleset(child.regime_filter, catalog, rng, config, strength)
    if rng.random() < 0.40:
        _mutate_exits(child.exit_rules, rng, strength)
    if rng.random() < 0.05:
        # Cambiar el lado operado es una mutacion drastica; se usa con cuentagotas.
        if rng.random() < 0.5:
            child.allow_long = not child.allow_long
        else:
            child.allow_short = not child.allow_short
        if not (child.allow_long or child.allow_short):
            child.allow_long = True

    _repair(child, catalog, rng, config)
    return child


def _mutate_ruleset(
    rules: RuleSet,
    catalog: FeatureCatalog,
    rng: np.random.Generator,
    config: EvolutionConfig,
    strength: float,
) -> None:
    action = rng.random()

    if action < 0.45 and rules.conditions:
        # Ajuste fino de un umbral: la mutacion mas util y menos destructiva.
        idx = int(rng.integers(len(rules.conditions)))
        condition = rules.conditions[idx]
        if condition.target_type == "const":
            spec = catalog.get(condition.feature)
            if spec is not None and spec.kind != "binary":
                span = (spec.high - spec.low) * 0.25 * max(strength, 0.05)
                condition.threshold = float(
                    np.clip(condition.threshold + rng.normal(0, span), spec.low - abs(spec.low), spec.high * 1.5)
                )
        else:
            condition.operator = OPERATORS[int(rng.integers(len(OPERATORS)))]

    elif action < 0.65 and rules.conditions:
        # Sustituir una condicion entera.
        idx = int(rng.integers(len(rules.conditions)))
        rules.conditions[idx] = random_condition(catalog, rng)

    elif action < 0.82 and len(rules.conditions) < config.max_conditions:
        rules.conditions.append(random_condition(catalog, rng))

    elif action < 0.92 and len(rules.conditions) > 1:
        # Podar: mantiene la complejidad a raya.
        rules.conditions.pop(int(rng.integers(len(rules.conditions))))

    else:
        rules.logic = str(rng.choice(LOGIC_MODES))


def _mutate_exits(rules: ExitRules, rng: np.random.Generator, strength: float) -> None:
    scale = max(strength, 0.05)
    choice = rng.random()

    if choice < 0.3:
        rules.stop_atr = float(np.clip(rules.stop_atr * np.exp(rng.normal(0, 0.25 * scale)), 0.3, 6.0))
    elif choice < 0.6:
        rules.target_atr = float(np.clip(rules.target_atr * np.exp(rng.normal(0, 0.25 * scale)), 0.3, 12.0))
    elif choice < 0.75:
        if rules.trail_atr > 0:
            rules.trail_atr = float(np.clip(rules.trail_atr * np.exp(rng.normal(0, 0.3 * scale)), 0.0, 6.0))
            if rules.trail_atr < 0.3:
                rules.trail_atr = 0.0
        else:
            rules.trail_atr = float(rng.uniform(0.5, 3.0))
    elif choice < 0.85:
        rules.breakeven_atr = 0.0 if rules.breakeven_atr > 0 else float(rng.uniform(0.5, 2.0))
    elif choice < 0.95:
        rules.max_bars = int(np.clip(rules.max_bars * np.exp(rng.normal(0, 0.35 * scale)), 6, 576))
    else:
        rules.exit_on_reverse = not rules.exit_on_reverse

    rules.validate()


def crossover(
    parent_a: StrategyGenome,
    parent_b: StrategyGenome,
    rng: np.random.Generator,
    config: EvolutionConfig,
    catalog: FeatureCatalog | None = None,
) -> tuple[StrategyGenome, StrategyGenome]:
    """Cruce uniforme por bloques funcionales.

    Se intercambian bloques completos (reglas de largo, de corto, filtro,
    salidas) en lugar de condiciones sueltas: un bloque coherente es la unidad
    con sentido, mezclar a nivel de condicion destruye lo aprendido.
    """
    child_a, child_b = parent_a.clone(), parent_b.clone()
    child_a.parents = child_b.parents = [parent_a.genome_id, parent_b.genome_id]
    child_a.generation = child_b.generation = max(parent_a.generation, parent_b.generation) + 1

    if rng.random() < 0.5:
        child_a.long_rules, child_b.long_rules = child_b.long_rules, child_a.long_rules
    if rng.random() < 0.5:
        child_a.short_rules, child_b.short_rules = child_b.short_rules, child_a.short_rules
    if rng.random() < 0.5:
        child_a.regime_filter, child_b.regime_filter = child_b.regime_filter, child_a.regime_filter
    if rng.random() < 0.5:
        child_a.exit_rules, child_b.exit_rules = child_b.exit_rules, child_a.exit_rules
    else:
        # Cruce aritmetico de los parametros de salida: explora el punto medio.
        alpha = float(rng.random())
        for attr in ("stop_atr", "target_atr", "trail_atr", "breakeven_atr"):
            va, vb = getattr(parent_a.exit_rules, attr), getattr(parent_b.exit_rules, attr)
            setattr(child_a.exit_rules, attr, alpha * va + (1 - alpha) * vb)
            setattr(child_b.exit_rules, attr, (1 - alpha) * va + alpha * vb)
        child_a.exit_rules.validate()
        child_b.exit_rules.validate()

    if rng.random() < 0.5:
        child_a.allow_long, child_b.allow_long = child_b.allow_long, child_a.allow_long
    if rng.random() < 0.5:
        child_a.allow_short, child_b.allow_short = child_b.allow_short, child_a.allow_short

    if catalog is not None:
        _repair(child_a, catalog, rng, config)
        _repair(child_b, catalog, rng, config)
    return child_a, child_b


def _repair(genome: StrategyGenome, catalog: FeatureCatalog, rng: np.random.Generator, config: EvolutionConfig) -> None:
    """Restaura invariantes que la mutacion o el cruce pueden haber roto.

    Sin esto aparecen genomas imposibles (operar largos sin reglas de largo,
    condiciones sobre features inexistentes) que ensucian la poblacion con
    individuos que nunca operan.
    """
    if not (genome.allow_long or genome.allow_short):
        genome.allow_long = True

    if genome.allow_long and not genome.long_rules.conditions:
        genome.long_rules = random_ruleset(catalog, rng, config.max_conditions)
    if genome.allow_short and not genome.short_rules.conditions:
        genome.short_rules = random_ruleset(catalog, rng, config.max_conditions)

    # Condiciones que apuntan a features ausentes del catalogo actual.
    for rules in (genome.long_rules, genome.short_rules, genome.regime_filter):
        for i, condition in enumerate(rules.conditions):
            invalid_feature = condition.feature not in catalog
            invalid_target = (
                condition.target_type == "feature"
                and (not condition.target_feature or condition.target_feature not in catalog)
            )
            if invalid_feature or invalid_target:
                rules.conditions[i] = random_condition(catalog, rng)
        if len(rules.conditions) > config.max_conditions:
            del rules.conditions[config.max_conditions :]

    genome.exit_rules.validate()
