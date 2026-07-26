"""GoldBot: sistema de trading autonomo para oro (XAU/USD) en temporalidad M5.

El sistema descubre sus propias estrategias mediante evolucion genetica,
las valida con walk-forward + Monte Carlo, las incuba en paper trading y
solo promueve a produccion aquellas que demuestran estabilidad sostenida.
Reaprende cada dia con los datos nuevos.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]
