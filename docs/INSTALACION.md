# Guía de instalación y arranque

Paso a paso, desde cero hasta el bot operando.

**Antes de empezar, dos cosas que determinan qué camino seguir:**

1. **MetaTrader 5 solo existe para Windows.** Si quieres operar XAU/USD o EUR/USD
   de verdad con XM o Vantage, necesitas Windows (o un VPS Windows). En Linux el
   bot funciona igual para descubrir y validar estrategias, pero no puede
   ejecutar órdenes en MT5.
2. **El bot arranca sin operar.** Viene con `dry_run: true` y no enviará una sola
   orden real hasta que lo desactives a mano. Déjalo así semanas.

| Tu situación | Camino |
|---|---|
| Windows, quiero operar con XM/Vantage | [Camino A](#camino-a-windows--mt5) |
| VPS Linux, solo descubrir estrategias | [Camino B](#camino-b-vps-linux-sin-mt5) |
| Quiero probarlo rápido sin instalar nada | [Camino C](#camino-c-docker) |

---

## Camino A: Windows + MT5

Es el único que permite operar oro y divisas de verdad.

**Resumen, si tienes prisa:** instala Python y MT5 → descarga el proyecto →
doble clic en `instalar.bat` → rellena el `.env` → doble clic en `arrancar.bat`.
Los dos `.bat` hacen todo lo demás. Los pasos de abajo son eso mismo, explicado.

### Paso 1 — Instalar Python

Descarga Python 3.11 o 3.12 desde [python.org](https://www.python.org/downloads/windows/).

> En el instalador, **marca la casilla "Add Python to PATH"**. Si se te olvida,
> ningún comando de esta guía funcionará y tendrás que reinstalar.

Comprueba que quedó bien abriendo PowerShell:

```powershell
python --version
```

Debe responder `Python 3.11.x` o superior.

### Paso 2 — Instalar y configurar MetaTrader 5

1. Descarga MT5 desde tu bróker: [XM](https://www.xm.com) o
   [Vantage](https://www.vantagemarkets.com).
2. Inicia sesión con tu cuenta (**usa primero una cuenta demo**).
3. En el terminal, ve a `Herramientas → Opciones → Asesores Expertos` y marca
   **"Permitir el trading automatizado"**. Sin esto, el bot podrá leer precios
   pero todas sus órdenes serán rechazadas.
4. **Importante para el histórico:** abre el gráfico del símbolo que vas a
   operar (`GOLD` en XM, `XAUUSD+` en Vantage), ponlo en **M5**, y desplázate
   hacia atrás con la rueda del ratón durante unos segundos. MT5 solo descarga
   el histórico que le pides mirando; sin esto, el bot recibirá 300 velas en
   lugar de 5.000.

### Paso 3 — Descargar el bot

> **Atención al punto final.** El repositorio se llama `Trading-aut-nomo.`,
> terminado en punto. Si escribes `Trading-aut-nomo.git`, git interpreta ese
> `.git` como sufijo y busca un repositorio sin el punto, que no existe:
> *Repository not found*. Las comillas evitan que la consola se coma el punto.

```powershell
cd %USERPROFILE%
git clone "https://github.com/josphermartinez-creator/Trading-aut-nomo." goldbot
cd goldbot
```

**Comprueba que funcionó antes de seguir:**

```powershell
dir requirements.txt
```

Si responde *No se encuentra el archivo*, el clon falló y los pasos siguientes
darán errores confusos (`No such file or directory: 'requirements.txt'`,
`neither setup.py nor pyproject.toml found`). Esos mensajes no significan que
falte Python: significan que estás en la carpeta equivocada.

**Alternativa sin git:** en la página del repositorio, botón verde **Code →
Download ZIP**. Descomprime y entra en la carpeta resultante.

### Paso 4 — Instalar

Dentro de la carpeta que acabas de descargar hay un fichero **`instalar.bat`**.
Haz doble clic en él. No hace falta abrir ninguna consola ni escribir nada.

Hace todo lo que antes había que teclear a mano: desactiva el proxy, comprueba
Python, crea el entorno virtual, instala las librerías, instala MetaTrader5 y
verifica que el bot arranca. Tarda unos minutos.

> **Por qué desactiva el proxy.** Si tienes una VPN o un antivirus que instala un
> proxy SOCKS, *todos* los comandos de pip fallan con `Missing dependencies for
> SOCKS support`. No se arregla instalando PySocks: para instalarlo pip necesita
> la red, que es justo lo que está bloqueado. La única salida es no usar el proxy
> mientras se instala, y eso es lo primero que hace `instalar.bat`. Solo afecta a
> esa ventana; no cambia nada en tu sistema.

Si termina con **`INSTALACION COMPLETADA`**, ya está. Si falla, la ventana te
dice qué pasó y qué hacer.

<details>
<summary>Los mismos pasos a mano, si prefieres la consola</summary>

```powershell
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m pip install MetaTrader5
.venv\Scripts\python.exe -m goldbot.cli --version
```

Se llama al intérprete del entorno virtual por ruta completa en vez de activarlo:
así no depende de que `activate` haya funcionado ni de cómo quede el `PATH`, que
es donde suele torcerse todo.

</details>

### Paso 5 — Configurar las credenciales

Al terminar, `instalar.bat` abre el Bloc de notas con el fichero `.env`. Rellena:

```ini
GOLDBOT_MODE=mt5
GOLDBOT_DRY_RUN=true          # NO lo cambies todavía

GOLDBOT_MT5_LOGIN=12345678    # tu número de cuenta
GOLDBOT_MT5_PASSWORD=tu_clave
GOLDBOT_MT5_SERVER=XMGlobal-Demo    # el que aparece en MT5
```

Guarda y cierra. **El bot lee este fichero solo, al arrancar**; no hay que cargar
nada a mano ni definir variables en la consola.

> El nombre del servidor está en MT5, abajo a la derecha, o en el correo de
> bienvenida del bróker. Tiene que coincidir **exactamente**.
>
> La contraseña es la **de la cuenta**, no la de inversor (*investor password*):
> esa última es de solo lectura y el bot no podría operar con ella.

Para volver a editarlo más adelante, usa la opción 7 de `arrancar.bat`.

### Paso 6 — Usar el bot: doble clic en `arrancar.bat`

En la misma carpeta está **`arrancar.bat`**. Ábrelo y verás un menú:

```
 1.  Descubrir estrategias   (obligatorio la primera vez)
 2.  Arrancar el bot         (opera + aprende cada día)
 3.  Ver el estado
 4.  Ver las estrategias encontradas
 5.  Informe de la estrategia campeona
 6.  Cambiar de instrumento  (oro / EURUSD)
 7.  Editar mis credenciales (.env)
 8.  Comprobar la conexión con MetaTrader 5
```

**El orden la primera vez es 8 → 1 → 4 → 2:** comprueba que el bot ve tu cuenta,
déjalo descubrir estrategias, mira qué encontró, y solo entonces arráncalo.

La opción 6 cambia entre oro y EUR/USD. Cada instrumento tiene su propia
configuración y su propio histórico, así que al cambiar hay que repetir el
paso 1.

El resto de esta sección explica lo mismo con comandos escritos a mano, por si
algo falla y quieres ver el detalle.

### Paso 7 — Comprobar que el bot ve tu cuenta

Es la opción 8 del menú. Con MT5 **abierto y con sesión iniciada**:

```powershell
.venv\Scripts\python.exe -m goldbot.cli data --update
```

Lo que debes buscar en la salida:

```
Simbolo resuelto para XAUUSD: GOLD (Gold vs US Dollar) | contrato 100 | ...
Descargadas 5000 velas 5m de GOLD [2025-11-14 22:00 -> 2026-01-08 06:45]
Costes leidos del broker: spread=0.35000 contrato=100 lote min=0.01
```

**Compara ese `spread` con el de tu `configs/default.yaml`.** Si el bróker te
cobra 0.35 y el fichero dice 0.25, ajústalo antes de seguir: en M5 el spread es
del mismo orden que el movimiento esperado, así que subestimarlo hace que
cualquier estrategia parezca rentable.

### Paso 8 — Descubrir estrategias

Es la opción 1 del menú.

```powershell
.venv\Scripts\python.exe -m goldbot.cli learn --bootstrap
```

**Tarda entre 30 y 120 minutos.** Descarga el histórico, hace evolucionar miles
de estrategias y valida las supervivientes. Déjalo correr sin tocar nada.

Al terminar verás algo así:

```
=== Ciclo de aprendizaje 2026-01-08 ===
Duracion: 2847s | barras: 5000
Descubiertas: 25 | validadas: 3
Campeon: ninguno
Decision: [SIN_CAMPEON] -: ningun retador listo todavia
```

> **"Validadas: 0" no es un error.** Significa que ninguna estrategia superó las
> cinco pruebas de estabilidad. Es un resultado legítimo y frecuente. No bajes
> los umbrales de `configs/default.yaml` para forzar que pase algo: eso solo
> hace que el bot opere con estrategias que ya sabías que no valían.

### Paso 9 — Revisar lo que encontró

Opciones 4 y 5 del menú.

```powershell
.venv\Scripts\python.exe -m goldbot.cli strategies              # lista todo
.venv\Scripts\python.exe -m goldbot.cli report champion         # las 5 pruebas
.venv\Scripts\python.exe -m goldbot.cli backtest champion --walkforward
```

### Paso 10 — Arrancar

Opción 2 del menú.

```powershell
.venv\Scripts\python.exe -m goldbot.cli schedule
```

Esto deja el bot funcionando: opera cada vela de 5 minutos y reaprende cada día
a las 22:00 UTC. **Con `dry_run=true` no envía órdenes reales** — incuba las
estrategias contra precios en vivo con dinero simulado.

Para que siga corriendo aunque cierres la sesión, déjalo en una ventana de
PowerShell abierta, o usa el Programador de tareas de Windows.

---

## Camino B: VPS Linux (sin MT5)

Sin MetaTrader 5 el bot **no puede operar**, pero sí descubrir y validar
estrategias usando datos de yfinance y CCXT. Sirve para dejar el descubrimiento
corriendo en un servidor barato mientras la ejecución vive en otro sitio.

### Instalación automática

```bash
sudo bash deploy/install_vps.sh
```

El script instala Python, crea el usuario de servicio, monta el entorno virtual
y registra el servicio systemd. Al terminar te dice qué hacer.

### Instalación manual

```bash
sudo apt update && sudo apt install -y python3 python3-venv git build-essential
git clone "https://github.com/josphermartinez-creator/Trading-aut-nomo." goldbot
cd goldbot
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

cp .env.example .env && nano .env        # rellena lo que necesites
set -a && source .env && set +a          # carga las variables

goldbot learn --bootstrap
goldbot status
```

Para dejarlo corriendo permanentemente:

```bash
sudo systemctl enable --now goldbot
sudo journalctl -u goldbot -f            # ver el log en directo
```

**Recomendación de máquina:** 2 vCPU y 4 GB. La evolución consume CPU durante
unos minutos al día; el resto del tiempo el bot duerme entre velas.

---

## Camino C: Docker

La forma más rápida de probarlo sin instalar nada en el sistema.

```bash
git clone "https://github.com/josphermartinez-creator/Trading-aut-nomo." goldbot
cd goldbot
cp .env.example .env        # edítalo si quieres Telegram
docker compose up -d
docker compose logs -f
```

Para ejecutar comandos dentro del contenedor:

```bash
docker compose exec goldbot goldbot status
docker compose exec goldbot goldbot learn --bootstrap
```

> Docker **no puede usar MetaTrader 5** (es Windows). Sirve para descubrimiento
> y papel, no para operar con XM o Vantage.

---

## Telegram (opcional pero muy recomendable)

Sin esto, el bot es una caja negra hasta que abres el terminal.

### Paso 1 — Crear el bot

1. En Telegram, busca **@BotFather**.
2. Envía `/newbot` y sigue las instrucciones.
3. Te dará un token parecido a `7123456789:AAF-abcdefgh...`. **Cópialo.**

### Paso 2 — Obtener tu chat_id

1. Busca tu bot recién creado y envíale cualquier mensaje (por ejemplo `hola`).
2. Abre en el navegador, sustituyendo `<TOKEN>` por el tuyo:

   ```
   https://api.telegram.org/bot<TOKEN>/getUpdates
   ```

3. Busca `"chat":{"id":123456789` — ese número es tu `chat_id`.

### Paso 3 — Configurar

En `.env`:

```ini
GOLDBOT_TELEGRAM_TOKEN=7123456789:AAF-abcdefgh...
GOLDBOT_TELEGRAM_CHAT_ID=123456789
```

Y en `configs/default.yaml` cambia `telegram: enabled: false` por `true`.

Al arrancar el bot recibirás un mensaje de confirmación. Comandos disponibles:

| Comando | Qué hace |
|---|---|
| `/estado` | Situación del bot y de la cuenta |
| `/posiciones` | Posiciones abiertas ahora |
| `/hoy` | Resultado del día |
| `/campeon` | La estrategia que está operando |
| `/estrategias` | Todas las descubiertas |
| `/pausar` | Deja de abrir posiciones nuevas |
| `/reanudar` | Vuelve a operar |
| `/cerrartodo` | Cierra todo *(pide confirmación)* |
| `/parar` | Detiene el bot *(pide confirmación)* |

---

## Operar EUR/USD

En Windows, la opción 6 de `arrancar.bat` cambia entre oro y EUR/USD. Desde la
consola es un proceso independiente por instrumento, con su propia configuración:

```bash
goldbot -c configs/eurusd.yaml learn --bootstrap
goldbot -c configs/eurusd.yaml schedule
```

Cada uno mantiene su caché, su base de datos y su campeón. **No los mezcles en
la misma configuración:** un motor evolutivo entrenado con oro y euro a la vez
produce estrategias promedio que no funcionan bien en ninguno.

---

## Pasar a dinero real

No hay prisa. El orden importa:

1. **Semanas en papel.** Deja el bot con `dry_run=true` hasta que tengas al menos
   un campeón con 50+ operaciones registradas.
2. **Compara papel contra backtest.** Si el rendimiento real es mucho peor que el
   simulado, el modelo de costes está mal calibrado — vuelve al Paso 6 y ajusta
   el spread. Esa divergencia es información valiosa, no un contratiempo.
3. **Cuenta demo primero**, aunque ya hayas incubado en papel. El bróker rechaza
   órdenes por motivos que ningún simulador reproduce.
4. **Cambia el interruptor**, en `.env`:

   ```ini
   GOLDBOT_DRY_RUN=false
   ```

5. **Empieza con el capital mínimo** que permita tu bróker.

El comando `goldbot run` te pedirá escribir literalmente `OPERAR EN REAL` antes
de continuar. Es a propósito.

---

## Problemas frecuentes

**`Repository not found` al clonar**
El nombre del repositorio termina en punto. Usa la URL entre comillas y **sin**
`.git`:
`git clone "https://github.com/josphermartinez-creator/Trading-aut-nomo." goldbot`

**`No such file or directory: 'requirements.txt'`**
No estás dentro de la carpeta del proyecto. Casi siempre porque el `git clone`
falló antes y el `cd goldbot` no llegó a funcionar. Vuelve al Paso 3.

**`neither 'setup.py' nor 'pyproject.toml' found`**
Lo mismo: `pip install -e .` se está ejecutando fuera del proyecto. Comprueba con
`dir requirements.txt` que estás donde debes.

**`Missing dependencies for SOCKS support`**
Tienes un proxy SOCKS configurado (lo dejan muchas VPN y algunos antivirus).
Desactívalo en esa ventana con `set HTTP_PROXY=`, `set HTTPS_PROXY=` y
`set ALL_PROXY=`, y repite el comando. **No intentes `pip install pysocks`:** para
instalarlo pip necesita la red, que es justo lo que el proxy está bloqueando.
`instalar.bat` y `arrancar.bat` ya hacen esto solos.

**`goldbot: command not found` / `no se reconoce como comando`**
No has activado el entorno virtual. En Windows lo más simple es no activarlo y
llamar al intérprete por ruta completa:
`.venv\Scripts\python.exe -m goldbot.cli status`. En Linux,
`source .venv/bin/activate`.

**Rellené el `.env` pero el bot sigue en modo `paper`**
El `.env` tiene que estar en la raíz del proyecto, junto a `requirements.txt`, y
la línea es `GOLDBOT_MODE=mt5` sin espacios alrededor del `=`. Ojo también con
Windows: si el Bloc de notas lo guardó como `.env.txt`, el bot no lo ve. Otra
causa: una variable definida en la consola gana sobre el fichero, a propósito.

**`La libreria MetaTrader5 no esta disponible`**
Solo existe en Windows. En Linux el bot seguirá funcionando con yfinance y CCXT,
pero sin poder operar.

**`No se encontro ningun simbolo para XAUUSD`**
El símbolo no está en Market Watch. En MT5, botón derecho sobre la lista de
símbolos → *Mostrar todo*, y busca `GOLD` o `XAUUSD`.

**`Se pidieron 5000 velas y solo llegaron 312`**
El terminal no tiene el histórico descargado. Abre el gráfico M5 del símbolo y
desplázate hacia atrás con la rueda del ratón. Luego repite la opción 8 del menú
(`goldbot data --update`).

**`costs.contract_size=100.0 no corresponde a EURUSD`**
Estás usando la configuración del oro para el euro. Usa `-c configs/eurusd.yaml`.
Este error es deliberado: sin él, tus posiciones serían mil veces mayores de lo
previsto.

**`Historico insuficiente. Se generan datos SINTETICOS`**
No hay datos reales suficientes. Los resultados **no sirven para operar** — es un
modo para que el sistema arranque en seco. Consigue histórico real antes de dar
validez a nada.

**El bot no abre ninguna operación**
Lo más probable es que sea el filtro de tendencia: con `method: combined` y
`allow_flat: false`, descarta en torno al 40% de las barras. Compruébalo con
`goldbot report champion`. Si quieres más actividad, `allow_flat: true` permite
operar en lateral **sin levantar la prohibición de ir contra tendencia**.

---

## Comprobación rápida

Si algo no encaja, esto te dice dónde estás:

```bash
goldbot status
```

```
=== ESTADO DE GOLDBOT ===
Modo ejecucion : mt5 (dry_run=True)
Datos          : 5.000 barras
                 2025-11-14 -> 2026-01-08
Campeon        : NINGUNO
En incubacion  : 3
Validadas      : 0 esperando hueco
Ultimo ciclo OK: 2026-01-08T22:04:11
```

- **Datos a 0** → el bróker no está entregando velas: revisa Paso 2 y Paso 6.
- **Campeón NINGUNO y 0 en incubación** → nada superó la puerta de estabilidad.
- **Campeón NINGUNO pero varias incubando** → normal, están cumpliendo sus 10
  días en papel.
