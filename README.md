# GoldBot — Bot autónomo de trading en M5 (XAU/USD y EUR/USD)

Sistema de trading que **inventa sus propias estrategias**, las valida con un
protocolo estadístico exigente, las incuba en papel y solo promueve a producción
aquellas que demuestran estabilidad sostenida. Reaprende cada día.

```
MT5 (XM/Vantage) ──► 5.000 velas reales ──► 114 features ──► Algoritmo genético
   autodetecta                              (incl. SMC)       (inventa reglas)
   el símbolo                                                        │
                                                    Optuna ◄─────────┘
                                                      │
   Trading en vivo ◄── Campeón ◄── Promoción ◄── Incubadora ◄── Puerta de
        │                                        papel 10d      estabilidad
        ├──► Telegram: avisos + /pausar /cerrartodo /parar
        └──► Vigilancia de deriva ──► Retirada si se degrada

        ⛔ VETO DE TENDENCIA — aplicado fuera del genoma, en toda señal
```

---

## ⚠️ Léelo antes de nada

**Este software no garantiza beneficios. Lo más probable es que pierdas dinero
operando en real.** El trading intradía de oro apalancado es una de las formas
más rápidas de arruinar una cuenta.

Lo que este sistema hace bien es **decirte la verdad**: está construido para
rechazar estrategias, no para encontrar ganadoras. Durante su desarrollo, el
algoritmo genético descubrió y explotó **tres fugas de información futura** en
el motor de backtesting (produciendo un Sharpe de 12 sobre datos puramente
aleatorios, algo imposible). Las tres están corregidas y cubiertas por tests
—ver [Honestidad del backtest](#honestidad-del-backtest)— pero la lección es la
que importa:

> Un optimizador suficientemente potente encuentra los errores de tu simulador
> antes que las ineficiencias del mercado.

Si el bot no encuentra ninguna estrategia que supere la puerta de estabilidad,
**ese es el resultado correcto**, no un fallo.

Arranca en `dry_run: true`. Déjalo así durante semanas.

---

## Instalación

> **Guía paso a paso completa:** [`docs/INSTALACION.md`](docs/INSTALACION.md)
> — cubre Windows + MT5 (XM/Vantage), VPS Linux, Docker, Telegram y el paso
> a dinero real.

```bash
git clone "https://github.com/josphermartinez-creator/Trading-aut-nomo."
cd Trading-aut-nomo
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
```

### Primeros pasos

```bash
goldbot data --update          # descarga y cachea el histórico M5
goldbot learn --bootstrap      # descubre estrategias (30–120 min)
goldbot strategies             # ¿qué encontró?
goldbot report champion        # informe detallado del campeón
goldbot backtest champion --walkforward
goldbot run                    # trading en vivo (papel por defecto)
```

En un VPS, todo en un proceso:

```bash
goldbot schedule               # trading + aprendizaje diario a las 22:00 UTC
```

---

## Cómo funciona


### 0. Arranque contra el bróker (XM / Vantage)

Al conectar con MetaTrader 5, lo primero que hace el bot es **descargar 5.000
velas M5 reales del bróker** y adaptar la estrategia sobre esos datos. Son
mejores que cualquier proxy: mismo spread, mismo horario de servidor, mismo
instrumento que vas a operar.

El símbolo **se autodetecta**, porque cada bróker lo llama de forma distinta:

| Bróker | Oro | Euro |
|---|---|---|
| **XM** | `GOLD` | `EURUSD` |
| **Vantage** (STP) | `XAUUSD+` | `EURUSD+` |
| otros | `XAUUSD.a`, `XAUUSDm`, `XAUUSD_i` | `EURUSD.a`, `EURUSDm` |

Si ninguno de los alias conocidos existe, el bot recorre el catálogo del bróker
buscando el instrumento. Y descarta cualquier símbolo cuyo precio caiga fuera
del rango esperado: si "XAUUSD" cotiza a 1,08, el símbolo se resolvió mal y
operar así es la forma más rápida de perderlo todo.

La vela en curso se descarta siempre — todavía se está formando.

### 0b. Dos instrumentos

```bash
goldbot -c configs/default.yaml learn --bootstrap   # oro
goldbot -c configs/eurusd.yaml  learn --bootstrap   # euro
```

Un proceso por instrumento, cada uno con su caché, su base de datos y su
campeón. Mezclarlos en un mismo motor evolutivo produciría estrategias promedio
que no funcionan bien en ninguno.

Lo único que cambia de verdad entre ambos es el **tamaño de contrato**: 100
onzas en oro frente a 100.000 unidades en euro. Con el mismo número de lotes, el
riesgo en dólares difiere en tres órdenes de magnitud. Por eso el sistema lo
impone desde el registro de instrumentos y **rechaza arrancar** si el YAML
declara un contrato que no corresponde:

```
ValueError: costs.contract_size=100.0 no corresponde a EURUSD
            (deberia ser 100000.0)
```

Ese error no es una molestia: es lo que impide que un descuido multiplique tus
posiciones por mil.

### 1. Datos — el histórico crece solo

El oro spot no cotiza en exchanges cripto, así que se combinan tres fuentes:

| Fuente | Instrumento | Alcance en M5 |
|---|---|---|
| **yfinance** | `GC=F` (futuros COMEX) | ~60 días (límite de Yahoo) |
| **CCXT** | `PAXG/USDT` (oro tokenizado 1:1) | Años, paginando |
| **CSV** | Export de MetaTrader 5 | El que tengas — mejor calidad |

Ningún proveedor gratuito entrega dos años de M5 de golpe. La solución es un
**caché incremental en Parquet**: cada ejecución diaria vuelca lo nuevo, y a los
pocos meses dispones de un histórico que no podrías descargar de una vez.

### 2. Features — 114 features causales

Tendencia, momento, volatilidad, volumen, estructura y sesión. Cada una se
registra con metadatos (`FeatureSpec`) que describen su *tipo* y su rango útil.

Eso es lo que permite al genético inventar reglas **con sentido**: comparar el
RSI con 70 es razonable, compararlo con el precio del oro no lo es. El catálogo
lo impide por construcción.

**Regla inviolable:** ningún indicador mira al futuro. Verificado por test.


### 2b. Smart Money Concepts y toma de liquidez

15 features SMC alimentan al genético junto al resto:

- **Estructura**: BOS (ruptura de estructura) y CHoCH (cambio de carácter)
- **Toma de liquidez**: barridos de stops sobre máximos/mínimos previos, y
  máximos/mínimos iguales (bolsas de liquidez en reposo)
- **Order blocks**: última vela opuesta antes del impulso que rompe estructura
- **Fair Value Gaps**: desequilibrios de tres velas sin negociación
- **Premium/discount**: posición dentro del rango operativo

**El detalle que casi todo el mundo se salta:** un swing no se conoce en su
propia vela. Hacen falta N velas posteriores que no lo superen para
confirmarlo. Las implementaciones que usan `rolling(center=True)` y marcan el
swing donde ocurrió meten N barras de futuro en cada señal — y sobre esa fuga el
genético construye estrategias irreproducibles.

Aquí la confirmación se desplaza siempre hacia atrás: en la barra `t` se
pregunta si la barra `t-right` fue el extremo de una ventana **que termina en
t**. Verificado por test.

### 2c. Prohibición de operar contra la tendencia

Esto **no es una condición del genoma**. Si lo fuera, el algoritmo la eliminaría
en cuanto encontrase un tramo del histórico donde ir a contracorriente rentase
más — y ese tramo siempre existe.

La dirección se calcula fuera del árbol genético, se inyecta como columna
reservada (`_trend_direction`) que **no aparece en el catálogo**, y se aplica
como veto final a toda señal. El genético no puede construir condiciones sobre
ella ni desactivarla.

| Método | Criterio |
|---|---|
| `ema_stack` | EMA 50 > 200 > 576 ordenadas |
| `ema_slope` | Pendiente de la EMA lenta |
| `structure` | Estructura de mercado SMC |
| `combined` | Acuerdo entre los tres *(por defecto)* |

Con `allow_flat: false` tampoco se opera en lateral: en un mercado sin dirección
la mejor operación suele ser ninguna. En mis pruebas eso deja fuera ~40% de las
barras — es el precio de la restricción, y es intencionado.

```
tests/test_smc_trend.py::test_ninguna_estrategia_puede_operar_contra_la_tendencia
30 estrategias aleatorias · 84.131 señales · 0 contra tendencia
```

### 2d. Telegram

Avisos de apertura y cierre de operaciones, informe diario, cambio de campeón y
cortacircuitos. Y control desde el móvil:

```
/estado      /posiciones   /hoy        /campeon
/estrategias /pausar       /reanudar
/cerrartodo  /parar        (piden confirmación)
```

Solo obedece al `chat_id` autorizado — el bot puede cerrar posiciones, así que
cualquier otro chat se registra y se ignora. El token va en
`GOLDBOT_TELEGRAM_TOKEN`, nunca en el YAML.

### 3. El genético — aquí es donde se inventan las estrategias

Cada estrategia se codifica como un genoma legible:

```
FILTRO DE RÉGIMEN: adx_14 > 22.4
LARGO SI:  ema_cross_9_50 > 0 Y rsi_14 < 68
CORTO SI:  ema_cross_9_50 cruza abajo 0
SALIDA:    stop 2.05xATR, objetivo 4.10xATR, trailing 2.5xATR, max 144 barras
```

Se eligió una representación plana en lugar de programación genética con árboles
arbitrarios por tres razones: el espacio de búsqueda queda acotado, toda
estrategia resultante se puede leer y auditar en castellano, y el sobreajuste es
mucho menor con pocos grados de libertad.

La población arranca sembrada con **9 arquetipos** clásicos (seguimiento de
tendencia, reversión a la media, ruptura, compresión de volatilidad, VWAP…) más
individuos aleatorios. El genético es libre de destrozarlos o descartarlos.

Salvaguardas: elitismo, nicho por huella estructural, inmigración aleatoria,
mutación adaptativa que repunta ante el estancamiento, y **validación en un
bloque reservado que la evolución nunca ve**.

### 4. La función de fitness — el fichero más delicado

El genético optimizará *exactamente* lo que se premie, incluidos los atajos no
previstos. Maximizar el retorno produce estrategias con tres operaciones
afortunadas; maximizar el Sharpe produce curvas con un drawdown que reventaría
la cuenta antes de recuperarse.

Por eso el fitness es multiplicativo y castiga:

- **Actividad mínima** — menos de 40 operaciones es una muestra sin valor
- **Drawdown** — factor que cae a 0 al alcanzar el límite tolerado
- **Concentración** — si 5 operaciones explican >60% del beneficio, fue suerte
- **Complejidad** — a igualdad de resultados gana la regla simple
- **Sobreoperación** — operar cada 5 minutos multiplica costes y fragilidad

### 5. La puerta de estabilidad — cinco pruebas independientes

Aquí es donde vive la exigencia de *"hasta conseguir la que se mantenga
estable"*. Basta fallar una prueba para quedar descartada:

| # | Prueba | Qué detecta |
|---|---|---|
| 1 | **Rendimiento absoluto** | Sharpe, PF, drawdown, nº de operaciones |
| 2 | **Consistencia walk-forward** | Gana en ≥65% de los tramos fuera de muestra |
| 3 | **Robustez Monte Carlo** | Reordenando operaciones, P(ruina) ≤ 5% |
| 4 | **Estabilidad paramétrica** | Conserva ≥60% del Sharpe al mover parámetros ±15% |
| 5 | **Regularidad temporal** | R² de la equity ≥0.55, ≥50% de meses en verde |

Son independientes a propósito: el sobreajuste supera una prueba por casualidad,
rara vez cinco.

### 6. La incubadora — el paso que nadie se salta

Ninguna estrategia llega a campeona sin operar **10 días en papel contra precios
reales**. Un backtest excelente y una ejecución real decepcionante son
perfectamente compatibles; la incubadora es lo único que los distingue antes de
que cueste dinero.

Reemplazar al campeón exige una **mejora ≥15%**, no una mejora cualquiera. Sin
ese margen el bot cambiaría de estrategia constantemente persiguiendo ruido.

### 7. Machine learning — meta-etiquetado, no predicción

Un error extendido es pedirle a un modelo que prediga la dirección del precio.
Con datos de 5 minutos eso produce AUC de 0.51 y modelos inútiles.

El enfoque aquí es distinto: **la dirección la decide el genoma**; el modelo solo
responde a *"dada esta señal concreta, ¿acabará ganando?"*. No toma posiciones,
**modula el tamaño**. Si el AUC fuera de muestra no supera 0.52, el modelo se
descarta: mejor ninguno que uno malo.

Etiquetado por **triple barrera** (López de Prado) con pesos por unicidad
temporal y splits **purgados con embargo** — sin eso, el AUC sale inflado.

### 8. Aprendizaje diario

Cada día a las 22:00 UTC (tras el cierre de NY):

1. Actualizar datos → el histórico crece
2. Recalcular features
3. **Vigilar deriva** → ¿cambió el régimen? ¿se degradó el campeón?
4. Descubrir (cada 3 días, o inmediatamente si hay deriva)
5. Refinar con Optuna
6. Puerta de estabilidad
7. Reentrenar el meta-modelo
8. Gestionar la incubadora
9. Promover / retirar

Cada etapa falla de forma aislada: si el descubrimiento revienta, el campeón
sigue operando. Un bot que se cae entero porque una etapa falló no es autónomo.

La detección de deriva distingue dos casos y reacciona distinto a cada uno:
**deriva de covariables** (el mercado cambió de régimen → reentrenar) frente a
**degradación de rendimiento** (la ventaja se agotó → retirar).

---

## Honestidad del backtest

El motor está construido con tres principios, en este orden:

1. **Ausencia de look-ahead.** La señal de la barra `t` se ejecuta en la apertura
   de `t+1`. El motor aplica ese desplazamiento internamente: una estrategia no
   puede hacer trampa aunque quiera.
2. **Pesimismo ante la ambigüedad.** Si dentro de una vela se tocan stop y
   objetivo, se asume el stop. Sin datos de tick no hay forma de saber el orden,
   y equivocarse hacia el lado optimista arruina cuentas reales.
3. **Velocidad.** ~80 ms por backtest sobre 30.000 barras.

### Las tres fugas que encontró el genético

Documentadas porque son errores que comete casi todo el mundo:

| Fuga | Síntoma | Corrección |
|---|---|---|
| El trailing stop se actualizaba con el máximo de la barra `i` pero no podía saltar hasta `i+1` | Una barra de subida gratis | El trailing se comprueba contra la misma vela que lo actualizó |
| La gestión de posición corría *antes* que la entrada | La vela de entrada nunca te saltaba el stop | Orden: salida por señal → entrada → gestión, todo dentro de la barra |
| Se usaba `ATR[i]` para fijar stops en `open[i]` | `ATR[i]` contiene el máximo y mínimo de esa vela | Se usa `ATR[i-1]` |

El síntoma que lo delató: **cientos de operaciones cerradas por `stop_loss` con
beneficio medio positivo**. Un stop que gana dinero es una contradicción.

Además, un **trailing mínimo de 0.5×ATR**: por debajo, la distancia es menor que
el ruido y la horquilla, y el "trailing" degenera en una orden de vender justo en
el máximo de la vela. Imposible de replicar en real. *El optimizador encuentra
siempre estos valores si se le deja.*

### Los tests que lo verifican

```bash
pytest tests/test_no_lookahead.py -v
```

- **Causalidad**: cada indicador calculado sobre un prefijo debe dar valores
  idénticos a los de la serie completa
- **Control negativo**: sobre un paseo aleatorio con costes, la estrategia
  mediana **debe perder dinero**. Si gana, hay una fuga
- **Control positivo**: una estrategia con acceso al futuro **debe ganar mucho**.
  Confirma que el motor sí permite ganar cuando hay ventaja real
- **Coherencia semántica**: una salida etiquetada `stop_loss` nunca puede tener
  P&L medio positivo

---

## Gestión de riesgo

Límites que el bot nunca sobrepasa (todos configurables en `configs/default.yaml`):

```yaml
risk_per_trade: 0.005       # 0.5% del capital por operación
max_daily_loss_pct: 0.03    # corta el día al -3%
max_drawdown_pct: 0.15      # se apaga al -15%
kelly_fraction: 0.25        # Kelly fraccional, nunca Kelly completo
max_trades_per_day: 20
```

El tamaño se deriva **siempre** de la distancia al stop. Sin stop no hay
operación. El Kelly se acota a [0.25, 1.5]: el objetivo es reducir tamaño en
malas rachas más que aumentarlo en las buenas, porque el Kelly estimado sobre
pocas muestras es muy optimista.

**Cortacircuitos** deliberadamente aburridos —la lógica que debe funcionar
cuando todo lo demás falla tiene que ser trivial de auditar—: drawdown máximo,
pérdida diaria, racha de pérdidas, errores repetidos del broker, precios
corruptos, huecos de precio y datos obsoletos.

---

## Ejecución real

| Modo | Instrumento | Cortos | Notas |
|---|---|---|---|
| `paper` | — | Sí | Por defecto. Precios reales, dinero simulado |
| `ccxt` | `PAXG/USDT` | **No** | Spot: tener PAXG *es* estar largo |
| `mt5` | `XAUUSD` / `EURUSD` | Sí | **La vía correcta** (XM, Vantage). Solo Windows |

**PAXG no es XAU/USD.** Cotiza contra USDT, tiene su propia prima y su liquidez
es una fracción de la del oro real. Para operar oro de verdad con cortos y
apalancamiento, MetaTrader 5 es el adaptador correcto.

### Activar dinero real

1. Semanas en `paper` con resultados consistentes
2. Comparar el rendimiento en papel con el del backtest — si divergen mucho, el
   modelo de costes está mal calibrado
3. `GOLDBOT_DRY_RUN=false` en `.env`
4. Empezar con el capital mínimo del broker

El comando `run` pide confirmación escrita antes de operar en real.

---

## Estructura

```
goldbot/
├── config.py           Configuración tipada (YAML + variables de entorno)
├── instruments.py      Registro XAUUSD / EURUSD (contrato, costes, rangos)
├── cli.py              Interfaz de línea de comandos
├── scheduler.py        Trading + aprendizaje diario en un proceso
├── data/               Proveedores (MT5, yfinance, CCXT, CSV), caché incremental
├── features/           114 features: indicadores, SMC, tendencia, etiquetado
├── strategies/         Genoma, operadores genéticos, arquetipos semilla
├── backtest/           Motor, costes, métricas, walk-forward, Monte Carlo
├── evolution/          Fitness, motor genético, refinamiento Optuna
├── ml/                 Meta-etiquetado, entrenamiento, deriva
├── risk/               Dimensionamiento, cortacircuitos
├── execution/          Broker base, papel, CCXT, MetaTrader 5
├── live/               Bucle de trading en vivo
├── autonomy/           Puerta de estabilidad, registro campeón, orquestador
├── notifications/      Bot de Telegram (avisos + control remoto)
└── storage/            Persistencia SQLite
```

---

## Despliegue en VPS

**Docker** (recomendado):

```bash
cp .env.example .env      # rellena credenciales
docker compose up -d
docker compose logs -f
```

**Nativo con systemd**:

```bash
sudo bash deploy/install_vps.sh
sudo -u goldbot /opt/goldbot/.venv/bin/goldbot learn --bootstrap
sudo systemctl start goldbot
```

Recomendación: 2 vCPU / 4 GB. La evolución es intensiva en CPU; el resto del
tiempo el bot duerme entre velas.

**Respaldo**: `data/goldbot.db` y `data/cache/` son el aprendizaje acumulado.
Cópialos periódicamente.

---

## Comandos

```bash
goldbot data --update                    # actualizar histórico
goldbot learn --bootstrap                # arranque en frío
goldbot learn                            # ciclo diario
goldbot learn --force-discovery          # forzar evolución
goldbot backtest champion --walkforward  # backtest detallado
goldbot strategies --status validated    # listar por estado
goldbot report <id>                      # informe con las 5 pruebas
goldbot status --json                    # estado del sistema
goldbot run --max-cycles 100             # trading en vivo
goldbot schedule                         # todo junto
```

---

## Herramientas utilizadas

| Uso | Herramienta |
|---|---|
| Lenguaje | Python 3.10+ |
| Datos históricos | yfinance, CCXT, CSV (MT5) |
| Backtesting | Motor vectorizado propio (~80 ms/backtest) |
| Machine Learning | scikit-learn (PyTorch opcional) |
| Notificaciones | API HTTP de Telegram (sin dependencias extra) |
| Optimización | Algoritmo genético propio + Optuna |
| Ejecución en vivo | CCXT, MetaTrader 5, broker de papel |
| Infraestructura | Docker / systemd en VPS |

Sobre el motor de backtesting propio: se evaluaron Backtrader, Zipline y
VectorBT. Backtrader y Zipline son orientados a eventos y demasiado lentos para
las miles de evaluaciones que exige el genético. VectorBT es rápido pero su
modelo vectorizado no expresa bien stops intrabarra con trailing dependiente del
camino —exactamente donde se esconden las fugas—. El motor propio hace ~80 ms por
backtest y, sobre todo, permite auditar línea a línea el orden de eventos dentro
de la vela. Están anotados en `requirements.txt` como opcionales para
verificación cruzada.

---

## Licencia y aviso legal

MIT. Software educativo y de investigación. No es asesoramiento financiero. El
trading apalancado puede acarrear pérdidas superiores al depósito inicial. Los
autores no se responsabilizan de pérdidas económicas derivadas de su uso.
