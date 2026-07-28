@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul 2>&1
title Instalador de GoldBot

REM ======================================================================
REM  GoldBot - instalador para Windows
REM
REM  Haz doble clic en este fichero. No hace falta estar en ninguna carpeta
REM  concreta: el script se situa solo en la del proyecto.
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
REM  1. Neutralizar el proxy en ESTA sesion
REM
REM  OJO con la parte contraintuitiva: NO basta con vaciar HTTP_PROXY y
REM  HTTPS_PROXY. En Windows, Python resuelve el proxy asi:
REM
REM      getproxies() = getproxies_environment() or getproxies_registry()
REM
REM  y getproxies_environment() descarta las variables vacias. Si las
REM  vaciamos, el diccionario queda vacio, eso es "falso", y Python pasa a
REM  leer el proxy DEL REGISTRO DE WINDOWS, que es justo donde lo dejan las
REM  VPN y los antivirus. Vaciar las variables no solo no arregla nada:
REM  provoca el problema.
REM
REM  La salida es dejar el diccionario no vacio pero sin proxy para http ni
REM  https. NO_PROXY tambien acaba en "_proxy", asi que cuenta como entrada
REM  y corta la consulta al registro.
REM
REM  Nada de esto toca tu sistema: solo afecta a esta ventana.
REM ----------------------------------------------------------------------
echo [1/7] Neutralizando el proxy para esta sesion...
set HTTP_PROXY=
set HTTPS_PROXY=
set ALL_PROXY=
set FTP_PROXY=
set PIP_PROXY=
set http_proxy=
set https_proxy=
set all_proxy=
set ftp_proxy=
set pip_proxy=
set NO_PROXY=*
set no_proxy=*
echo       hecho.
echo.

REM ----------------------------------------------------------------------
REM  2. Elegir el interprete
REM
REM  Se prefiere 3.12 o 3.13 sobre las mas nuevas: MetaTrader5 publica sus
REM  wheels con retraso y en la version recien salida suele no haber. Sin
REM  MetaTrader5 no hay conexion con XM ni con Vantage.
REM ----------------------------------------------------------------------
echo [2/7] Buscando un Python compatible...
set "PY="
call :probar_python 3.12
if not defined PY call :probar_python 3.13
if not defined PY call :probar_python 3.11
if not defined PY call :probar_python 3.10
if not defined PY (
    python -c "import sys" >nul 2>&1
    if not errorlevel 1 set "PY=python"
)
if not defined PY (
    echo.
    echo   ERROR: Python no esta instalado, o no esta en el PATH.
    echo.
    echo   Descargalo de https://www.python.org/downloads/windows/
    echo   y en el instalador MARCA la casilla "Add Python to PATH".
    echo.
    echo   Recomendado: Python 3.12, que es el que mejor soporte tiene
    echo   para MetaTrader5.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%v in ('%PY% --version 2^>^&1') do set "PYVER=%%v"
echo       Usando Python !PYVER!  ^(%PY%^)

set "PY_NUEVO="
echo !PYVER! | findstr /b /c:"3.14" /c:"3.15" /c:"3.16" >nul && set "PY_NUEVO=1"
if defined PY_NUEVO (
    echo.
    echo       AVISO: Python !PYVER! es muy reciente y MetaTrader5 puede que
    echo       todavia no tenga version para el. Si mas adelante falla ese
    echo       paso, instala Python 3.12 desde python.org y vuelve a
    echo       ejecutar este script: lo detectara y lo usara.
)
echo.

REM ----------------------------------------------------------------------
REM  3. Comprobar que estamos en el proyecto
REM ----------------------------------------------------------------------
echo [3/7] Comprobando el proyecto...
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
REM  4. Diagnostico de red ANTES de instalar
REM
REM  Sin esto, cualquier fallo de pip queda como "no se pudo descargar" y
REM  hay que adivinar la causa. Esta comprobacion la dice.
REM ----------------------------------------------------------------------
echo [4/7] Comprobando la salida a internet...
echo.
%PY% scripts\diagnostico_red.py
set "RED=!errorlevel!"
echo.
if not "!RED!"=="0" (
    echo ======================================================================
    echo   NO HAY CONEXION UTILIZABLE. Arriba tienes el motivo y la solucion.
    echo ======================================================================
    echo.
    pause
    exit /b 1
)

REM ----------------------------------------------------------------------
REM  5. Entorno virtual
REM
REM  Si ya existe pero se creo con otra version de Python, se rehace: un
REM  venv de 3.14 con un interprete base 3.12 da errores incomprensibles.
REM ----------------------------------------------------------------------
echo [5/7] Preparando el entorno virtual...
set "VPY=%CD%\.venv\Scripts\python.exe"

if exist "!VPY!" (
    set "VENVVER="
    for /f "tokens=2" %%v in ('"!VPY!" --version 2^>^&1') do set "VENVVER=%%v"
    if "!VENVVER!"=="!PYVER!" (
        echo       Ya existe con Python !VENVVER!, se reutiliza.
    ) else (
        echo       Existe uno de Python !VENVVER! pero vamos a usar !PYVER!.
        echo       Se rehace para evitar mezclas.
        rmdir /s /q ".venv"
    )
)

if not exist "!VPY!" (
    %PY% -m venv .venv
    if errorlevel 1 (
        echo   ERROR: no se pudo crear el entorno virtual.
        pause
        exit /b 1
    )
    echo       Creado.
)

if not exist "!VPY!" (
    echo   ERROR: el entorno virtual esta corrupto. Borra la carpeta .venv
    echo   y vuelve a ejecutar este script.
    pause
    exit /b 1
)
echo.

REM ----------------------------------------------------------------------
REM  6. Instalar dependencias
REM
REM  Se llama al python del entorno virtual por ruta completa en lugar de
REM  activarlo: asi no dependemos de que "activate" haya funcionado ni de
REM  como quede el PATH, que es donde suele torcerse todo.
REM ----------------------------------------------------------------------
echo [6/7] Instalando dependencias. Esto tarda varios minutos...
echo.

echo       - actualizando pip
call :instalar --upgrade pip
if errorlevel 1 goto :error_red

echo       - librerias principales ^(numpy, pandas, scikit-learn...^)
call :instalar -r requirements.txt
if errorlevel 1 goto :error_red

echo       - el propio bot
call :instalar -e .
if errorlevel 1 goto :error_red

echo       - MetaTrader 5
call :instalar MetaTrader5
if errorlevel 1 (
    echo.
    echo       AVISO: MetaTrader5 no se pudo instalar.
    if defined PY_NUEVO (
        echo       Casi seguro es por Python !PYVER!, demasiado reciente para
        echo       esa libreria. Instala Python 3.12 desde python.org y vuelve
        echo       a ejecutar este script; lo detectara solo.
    ) else (
        echo       Reintentalo despues con:
        echo         .venv\Scripts\python.exe -m pip install MetaTrader5
    )
    echo.
    echo       El bot funcionara igual para descubrir estrategias, pero no
    echo       podra conectarse a XM ni a Vantage hasta que esto se resuelva.
    echo.
)
echo.

REM ----------------------------------------------------------------------
REM  7. Verificar
REM ----------------------------------------------------------------------
echo [7/7] Verificando la instalacion...
"!VPY!" -c "import goldbot, numpy, pandas, sklearn; print('       modulos: OK')"
if errorlevel 1 goto :error_import

"!VPY!" -m goldbot.cli --version
if errorlevel 1 goto :error_import

"!VPY!" -c "import MetaTrader5; print('       MetaTrader5: OK')" 2>nul
if errorlevel 1 echo       MetaTrader5: NO disponible ^(solo afecta a operar en real^)

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

REM ======================================================================
REM  Subrutinas
REM ======================================================================

REM Comprueba si existe una version concreta a traves del lanzador "py".
:probar_python
py -%1 -c "import sys" >nul 2>&1
if not errorlevel 1 set "PY=py -%1"
goto :eof

REM Instala con pip. Si falla, reintenta con --isolated, que ignora el
REM fichero pip.ini del usuario: un "proxy = socks5://..." escrito ahi no se
REM ve en las variables de entorno y sobrevive a todo lo anterior.
:instalar
"!VPY!" -m pip install --quiet --retries 3 --timeout 30 %*
if not errorlevel 1 goto :eof
echo         reintentando sin la configuracion de pip del usuario...
"!VPY!" -m pip install --quiet --isolated --retries 3 --timeout 30 %*
goto :eof

REM ----------------------------------------------------------------------
:error_red
echo.
echo ======================================================================
echo   ERROR AL DESCARGAR LOS PAQUETES
echo ======================================================================
echo.
echo   La comprobacion de red del paso 4 habia pasado, asi que internet
echo   funciona. Lo mas probable ahora es una de estas:
echo.
echo   a^) Un paquete no tiene version para Python !PYVER!. Instala
echo      Python 3.12 desde python.org y vuelve a ejecutar este script.
echo.
echo   b^) La descarga se corto a medias. Vuelve a ejecutarlo: pip reanuda
echo      desde donde estaba y no repite lo ya instalado.
echo.
echo   c^) El antivirus bloqueo la escritura en la carpeta .venv. Anade
echo      esta carpeta a sus excepciones.
echo.
echo   Copia el texto de arriba y mandalo si sigue fallando.
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
