@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1

REM ======================================================================
REM  GoldBot - instalador para Windows
REM
REM  Haz doble clic en este fichero, o ejecutalo desde la consola.
REM  No hace falta estar en ninguna carpeta concreta: el script se situa
REM  solo en la del proyecto.
REM ======================================================================

cd /d "%~dp0"

echo.
echo ======================================================================
echo   INSTALADOR DE GOLDBOT
echo ======================================================================
echo.
echo   Carpeta: %CD%
echo.

REM ----------------------------------------------------------------------
REM  1. Desactivar el proxy en ESTA sesion
REM
REM  Si hay un proxy SOCKS configurado (lo dejan muchas VPN y algunos
REM  antivirus), pip falla con "Missing dependencies for SOCKS support" en
REM  TODOS los comandos. No se arregla instalando PySocks, porque para
REM  instalarlo pip necesita la red, que es justo lo que esta bloqueado.
REM  La solucion es no usar el proxy mientras instalamos.
REM
REM  Esto solo afecta a esta ventana. No cambia nada en tu sistema.
REM ----------------------------------------------------------------------
echo [1/6] Desactivando proxy para esta sesion...
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=
echo       hecho.
echo.

REM ----------------------------------------------------------------------
REM  2. Comprobar Python
REM ----------------------------------------------------------------------
echo [2/6] Comprobando Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo   ERROR: Python no esta instalado, o no esta en el PATH.
    echo.
    echo   Descargalo de https://www.python.org/downloads/windows/
    echo   y en el instalador MARCA la casilla "Add Python to PATH".
    echo.
    pause
    exit /b 1
)
for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo       Python !PYVER! encontrado.
echo.

REM ----------------------------------------------------------------------
REM  3. Comprobar que estamos en el proyecto
REM ----------------------------------------------------------------------
echo [3/6] Comprobando el proyecto...
if not exist "requirements.txt" (
    echo.
    echo   ERROR: no encuentro requirements.txt en esta carpeta.
    echo.
    echo   Este script tiene que estar DENTRO de la carpeta del proyecto,
    echo   junto a requirements.txt y pyproject.toml.
    echo.
    pause
    exit /b 1
)
if not exist "pyproject.toml" (
    echo   ERROR: falta pyproject.toml. La descarga esta incompleta.
    pause
    exit /b 1
)
echo       requirements.txt y pyproject.toml encontrados.
echo.

REM ----------------------------------------------------------------------
REM  4. Entorno virtual
REM ----------------------------------------------------------------------
echo [4/6] Preparando el entorno virtual...
if exist ".venv\Scripts\python.exe" (
    echo       Ya existe, se reutiliza.
) else (
    python -m venv .venv
    if errorlevel 1 (
        echo   ERROR: no se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo       Creado.
)

set "VPY=%CD%\.venv\Scripts\python.exe"
if not exist "!VPY!" (
    echo   ERROR: el entorno virtual esta corrupto. Borra la carpeta .venv
    echo   y vuelve a ejecutar este script.
    pause
    exit /b 1
)
echo.

REM ----------------------------------------------------------------------
REM  5. Instalar dependencias
REM
REM  Se llama al python del entorno virtual por ruta completa en lugar de
REM  activarlo: asi no dependemos de que "activate" haya funcionado ni de
REM  que el PATH quede bien, que es donde suele torcerse todo.
REM ----------------------------------------------------------------------
echo [5/6] Instalando dependencias. Esto tarda varios minutos...
echo.

echo       - actualizando pip
"!VPY!" -m pip install --quiet --upgrade pip
if errorlevel 1 goto :error_red

echo       - librerias principales (numpy, pandas, scikit-learn...)
"!VPY!" -m pip install --quiet -r requirements.txt
if errorlevel 1 goto :error_red

echo       - el propio bot
"!VPY!" -m pip install --quiet -e .
if errorlevel 1 goto :error_red

echo       - MetaTrader 5
"!VPY!" -m pip install --quiet MetaTrader5
if errorlevel 1 (
    echo.
    echo       AVISO: MetaTrader5 no se pudo instalar.
    echo       El bot funcionara para descubrir estrategias, pero no podra
    echo       conectarse a XM ni a Vantage. Reintentalo despues con:
    echo         .venv\Scripts\python.exe -m pip install MetaTrader5
    echo.
)
echo.

REM ----------------------------------------------------------------------
REM  6. Verificar
REM ----------------------------------------------------------------------
echo [6/6] Verificando la instalacion...
"!VPY!" -c "import goldbot, numpy, pandas, sklearn; print('       modulos: OK')"
if errorlevel 1 goto :error_import

"!VPY!" -m goldbot.cli --version
if errorlevel 1 goto :error_import

"!VPY!" -c "import MetaTrader5; print('       MetaTrader5: OK')" 2>nul
if errorlevel 1 echo       MetaTrader5: NO disponible ^(solo afecta a operar en real^)

REM Fichero de credenciales
if not exist ".env" (
    if exist ".env.example" (
        copy /y ".env.example" ".env" >nul
        echo       .env creado desde la plantilla.
    )
)

echo.
echo ======================================================================
echo   INSTALACION COMPLETADA
echo ======================================================================
echo.
echo   SIGUIENTE PASO: pon tus credenciales de MetaTrader 5.
echo.
echo     1. Al cerrar este mensaje se abre el Bloc de notas con el .env
echo     2. Rellena GOLDBOT_MT5_LOGIN, _PASSWORD y _SERVER
echo     3. Pon tambien  GOLDBOT_MODE=mt5
echo     4. Guarda y cierra el Bloc de notas
echo.
echo   A partir de ahora, para usar el bot solo tienes que abrir
echo   "arrancar.bat" (esta en esta misma carpeta). Este instalador no
echo   hace falta volver a ejecutarlo.
echo.
echo ======================================================================
echo.
pause
notepad .env

echo.
set "AHORA="
set /p "AHORA=  Abrir el bot ahora? (S/N): "
if /i "!AHORA!"=="S" if exist "arrancar.bat" start "" "arrancar.bat"
exit /b 0

REM ----------------------------------------------------------------------
:error_red
echo.
echo ======================================================================
echo   ERROR AL DESCARGAR LOS PAQUETES
echo ======================================================================
echo.
echo   Lo mas probable es una de estas dos:
echo.
echo   a^) No hay conexion a internet ahora mismo.
echo.
echo   b^) Tu red OBLIGA a pasar por un proxy. Este script lo desactivo
echo      para poder instalar, pero si tu conexion no funciona sin el,
echo      hara falta configurarlo. Prueba a desconectar la VPN si tienes
echo      una activa, y vuelve a ejecutar este script.
echo.
pause
exit /b 1

:error_import
echo.
echo ======================================================================
echo   LA INSTALACION TERMINO PERO EL BOT NO ARRANCA
echo ======================================================================
echo.
echo   Borra la carpeta .venv y vuelve a ejecutar este script.
echo   Si sigue fallando, copia todo el texto de esta ventana y mandalo.
echo.
pause
exit /b 1
