@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
title GoldBot

REM ======================================================================
REM  GoldBot - lanzador para Windows
REM
REM  Haz doble clic. Antes tiene que haberse ejecutado "instalar.bat"
REM  al menos una vez.
REM ======================================================================

cd /d "%~dp0"

REM El proxy SOCKS que bloquea a pip tambien bloquea las descargas de
REM historico. Se desactiva solo para esta ventana.
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=

set "VPY=%CD%\.venv\Scripts\python.exe"
if not exist "!VPY!" (
    echo.
    echo   No encuentro el entorno virtual.
    echo   Ejecuta primero "instalar.bat" y vuelve a intentarlo.
    echo.
    pause
    exit /b 1
)

if not exist ".env" (
    if exist ".env.example" copy /y ".env.example" ".env" >nul
)

REM Instrumento por defecto: oro.
set "INSTRUMENTO=XAUUSD (oro)"
set "CFG=configs\default.yaml"

REM ----------------------------------------------------------------------
:menu
cls
echo ======================================================================
echo   GOLDBOT
echo ======================================================================
echo.
echo   Instrumento actual : !INSTRUMENTO!
echo   Configuracion      : !CFG!
echo.
echo   ------------------------------------------------------------------
echo    1.  Descubrir estrategias   (obligatorio la primera vez)
echo    2.  Arrancar el bot         (opera + aprende cada dia)
echo    3.  Ver el estado
echo    4.  Ver las estrategias encontradas
echo    5.  Informe de la estrategia campeona
echo   ------------------------------------------------------------------
echo    6.  Cambiar de instrumento  (oro / EURUSD)
echo    7.  Editar mis credenciales (.env)
echo    8.  Comprobar la conexion con MetaTrader 5
echo   ------------------------------------------------------------------
echo    0.  Salir
echo.
set "OPCION="
set /p "OPCION=  Elige una opcion y pulsa Enter: "

if "!OPCION!"=="1" goto :descubrir
if "!OPCION!"=="2" goto :operar
if "!OPCION!"=="3" goto :estado
if "!OPCION!"=="4" goto :estrategias
if "!OPCION!"=="5" goto :informe
if "!OPCION!"=="6" goto :instrumento
if "!OPCION!"=="7" goto :credenciales
if "!OPCION!"=="8" goto :probar_mt5
if "!OPCION!"=="0" exit /b 0
goto :menu

REM ----------------------------------------------------------------------
:descubrir
cls
echo ======================================================================
echo   DESCUBRIR ESTRATEGIAS  -  !INSTRUMENTO!
echo ======================================================================
echo.
echo   El bot va a:
echo     1. Descargar 5000 velas M5 reales de tu broker
echo     2. Hacer evolucionar miles de estrategias
echo     3. Quedarse solo con las que pasan las pruebas de estabilidad
echo.
echo   Tarda entre 30 y 120 minutos. Puedes dejarlo trabajando.
echo.
echo   ANTES DE SEGUIR, comprueba que:
echo     - MetaTrader 5 esta ABIERTO y con la sesion iniciada
echo     - El boton "AutoTrading" esta en verde
echo     - Has abierto un grafico M5 del simbolo y has subido hacia atras
echo       con la rueda del raton (si no, el broker solo entrega ~300 velas)
echo.
pause
echo.
"!VPY!" -m goldbot.cli --config "!CFG!" learn --bootstrap
echo.
if errorlevel 1 (
    echo   Termino con errores. Revisa el texto de arriba.
) else (
    echo   Listo. Mira el resultado con la opcion 4 del menu.
)
echo.
pause
goto :menu

REM ----------------------------------------------------------------------
:operar
cls
echo ======================================================================
echo   ARRANCAR EL BOT  -  !INSTRUMENTO!
echo ======================================================================
echo.
echo   Arranca el ciclo continuo: opera segun la estrategia campeona y
echo   cada dia vuelve a aprender con las velas nuevas.
echo.
echo   De fabrica viene en modo SIMULACION (dry_run). No manda ordenes
echo   reales al broker. Dejalo asi varias semanas y revisa resultados
echo   antes de plantearte cambiarlo.
echo.
echo   Para pararlo: pulsa Ctrl+C en esta ventana.
echo.
pause
echo.
"!VPY!" -m goldbot.cli --config "!CFG!" schedule
echo.
echo   El bot se ha detenido.
echo.
pause
goto :menu

REM ----------------------------------------------------------------------
:estado
cls
"!VPY!" -m goldbot.cli --config "!CFG!" status
echo.
pause
goto :menu

REM ----------------------------------------------------------------------
:estrategias
cls
"!VPY!" -m goldbot.cli --config "!CFG!" strategies
echo.
pause
goto :menu

REM ----------------------------------------------------------------------
:informe
cls
"!VPY!" -m goldbot.cli --config "!CFG!" report champion
echo.
pause
goto :menu

REM ----------------------------------------------------------------------
:instrumento
cls
echo ======================================================================
echo   ELEGIR INSTRUMENTO
echo ======================================================================
echo.
echo    1.  XAUUSD  (oro)
echo    2.  EURUSD
echo.
set "SEL="
set /p "SEL=  Opcion: "
if "!SEL!"=="1" (
    set "INSTRUMENTO=XAUUSD (oro)"
    set "CFG=configs\default.yaml"
)
if "!SEL!"=="2" (
    set "INSTRUMENTO=EURUSD"
    set "CFG=configs\eurusd.yaml"
)
goto :menu

REM ----------------------------------------------------------------------
:credenciales
cls
echo ======================================================================
echo   CREDENCIALES
echo ======================================================================
echo.
echo   Se abre el Bloc de notas con el fichero .env.
echo.
echo   Para operar con XM o Vantage rellena:
echo     GOLDBOT_MT5_LOGIN     numero de cuenta
echo     GOLDBOT_MT5_PASSWORD  contrasena de la cuenta (no la de inversor)
echo     GOLDBOT_MT5_SERVER    servidor exacto, tal cual aparece en MT5
echo     GOLDBOT_MODE=mt5
echo.
echo   Para recibir avisos por Telegram:
echo     GOLDBOT_TELEGRAM_TOKEN    te lo da @BotFather
echo     GOLDBOT_TELEGRAM_CHAT_ID  tu id de chat
echo.
echo   Guarda y cierra el Bloc de notas para volver al menu.
echo.
pause
notepad .env
goto :menu

REM ----------------------------------------------------------------------
:probar_mt5
cls
echo ======================================================================
echo   COMPROBANDO LA CONEXION CON METATRADER 5
echo ======================================================================
echo.
"!VPY!" -m goldbot.cli --config "!CFG!" data --update
echo.
if errorlevel 1 (
    echo   No se pudo descargar el historico.
    echo.
    echo   Comprueba en este orden:
    echo     - MetaTrader 5 abierto y con la sesion iniciada
    echo     - AutoTrading en verde
    echo     - Credenciales correctas en .env ^(opcion 7^)
    echo     - GOLDBOT_MODE=mt5 en el fichero .env
)
echo.
pause
goto :menu
