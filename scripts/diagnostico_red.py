"""Diagnostico de red: por que pip no puede descargar nada.

Se ejecuta antes de instalar. Devuelve 0 si hay salida a PyPI y un codigo
distinto de cero, con una explicacion concreta, si no la hay.

El caso que motiva este fichero
-------------------------------
En Windows, ``urllib.request.getproxies()`` es::

    def getproxies():
        return getproxies_environment() or getproxies_registry()

Y ``getproxies_environment()`` ignora las variables vacias. Es decir: vaciar
HTTP_PROXY y HTTPS_PROXY no desactiva el proxy, sino que hace que Python
**caiga al proxy del registro de Windows**, que es justamente donde lo dejan
las VPN y los antivirus. Si ese proxy es SOCKS, pip muere con
"Missing dependencies for SOCKS support" y da igual cuantas veces vacies las
variables.

La forma de cortar esa cadena sin tocar el sistema es dejar el diccionario de
entorno *no vacio* pero sin proxy para http/https: basta con NO_PROXY, que
tambien acaba en ``_proxy`` y por tanto cuenta como entrada. Con eso
``getproxies_environment()`` devuelve algo, el registro no se consulta, y no
hay proxy para https: conexion directa.
"""

from __future__ import annotations

import os
import socket
import ssl
import sys
import urllib.error
import urllib.request

PYPI_HOST = "pypi.org"
PYPI_URL = "https://pypi.org/simple/pip/"
TIMEOUT = 15

OK = 0
ERR_PROXY_SOCKS = 3
ERR_PROXY_OTRO = 4
ERR_SIN_DNS = 5
ERR_SIN_CONEXION = 6
ERR_TLS = 7


def neutralizar_proxy() -> None:
    """Deja el entorno de forma que Python no use ningun proxy.

    Importante el orden: primero se vacian las variables de proxy reales y
    despues se define NO_PROXY. Si solo se vaciaran, Windows caeria al
    registro (ver docstring del modulo).
    """
    for nombre in (
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY", "PIP_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "pip_proxy",
    ):
        os.environ.pop(nombre, None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"


# Los unicos esquemas que pip usa. Cualquier otra variable acabada en
# "_proxy" (npm_config_https_proxy y demas) aparece en getproxies() pero es
# ruido: nadie la consulta para una descarga https.
ESQUEMAS_RELEVANTES = ("https", "http", "all")


def proxies_visibles() -> dict[str, str]:
    """Proxies que pip usaria de verdad para bajar un paquete."""
    todos = urllib.request.getproxies()
    return {k: v for k, v in todos.items() if k in ESQUEMAS_RELEVANTES}


def proxy_de_pip_ini() -> str | None:
    """Un ``proxy =`` en pip.ini gana sobre todo lo demas y no se ve en el entorno."""
    candidatos = []
    if appdata := os.environ.get("APPDATA"):
        candidatos.append(os.path.join(appdata, "pip", "pip.ini"))
    if userprofile := os.environ.get("USERPROFILE"):
        candidatos.append(os.path.join(userprofile, "pip", "pip.ini"))
    candidatos.append(os.path.expanduser("~/.config/pip/pip.conf"))
    candidatos.append(os.path.expanduser("~/.pip/pip.conf"))

    for ruta in candidatos:
        try:
            with open(ruta, encoding="utf-8", errors="replace") as fh:
                for linea in fh:
                    limpia = linea.strip()
                    if limpia.lower().startswith("proxy") and "=" in limpia:
                        valor = limpia.split("=", 1)[1].strip()
                        if valor:
                            return f"{ruta} -> proxy = {valor}"
        except OSError:
            continue
    return None


def probar_pypi() -> tuple[int, str]:
    """Intenta hablar con PyPI de verdad. Es la unica prueba que vale."""
    try:
        socket.getaddrinfo(PYPI_HOST, 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return ERR_SIN_DNS, f"No se resuelve {PYPI_HOST} ({exc})"

    peticion = urllib.request.Request(PYPI_URL, headers={"User-Agent": "goldbot-diagnostico"})
    try:
        with urllib.request.urlopen(peticion, timeout=TIMEOUT) as resp:
            resp.read(64)
        return OK, "Conexion con PyPI correcta."
    except urllib.error.URLError as exc:
        motivo = exc.reason
        texto = str(motivo)
        if "SOCKS" in texto:
            return ERR_PROXY_SOCKS, texto
        if isinstance(motivo, ssl.SSLError):
            return ERR_TLS, texto
        return ERR_SIN_CONEXION, texto
    except Exception as exc:  # noqa: BLE001 - aqui cualquier fallo es informacion
        texto = str(exc)
        if "SOCKS" in texto:
            return ERR_PROXY_SOCKS, texto
        return ERR_SIN_CONEXION, texto


def afinar_codigo(codigo: int, proxy_persiste: bool, pip_ini: str | None) -> int:
    """Un fallo generico habiendo un proxy que sobrevivio no es "no hay internet".

    ``probar_pypi`` solo ve la excepcion, y muchas veces es un timeout pelado
    sin la palabra SOCKS. Si ademas sabemos que quedaba un proxy en pie, el
    consejo util no es "comprueba el cable" sino "quita ese proxy".
    """
    if codigo == ERR_SIN_CONEXION and (proxy_persiste or pip_ini):
        return ERR_PROXY_OTRO
    return codigo


def main() -> int:
    print(f"  Python      : {sys.version.split()[0]} ({sys.executable})")

    ini = proxy_de_pip_ini()
    if ini:
        print(f"  pip.ini     : {ini}")

    antes = proxies_visibles()
    proxy_persiste = False
    if antes:
        print(f"  Proxy activo: {antes}")
        neutralizar_proxy()
        despues = proxies_visibles()
        if despues:
            proxy_persiste = True
            print(f"  Tras limpiar: {despues}  <-- no se ha podido neutralizar")
        else:
            print("  Tras limpiar: ninguno (conexion directa)")
    else:
        neutralizar_proxy()
        print("  Proxy activo: ninguno (conexion directa)")

    codigo, detalle = probar_pypi()
    print(f"  PyPI        : {detalle}")

    codigo = afinar_codigo(codigo, proxy_persiste, ini)

    if codigo == OK:
        return OK

    print()
    if codigo == ERR_PROXY_SOCKS:
        print("  DIAGNOSTICO: hay un proxy SOCKS que Python no sabe usar y que no")
        print("  se ha podido desactivar desde aqui. Casi siempre viene de una VPN")
        print("  o de un antivirus, y esta puesto en la configuracion de Windows,")
        print("  no en una variable de entorno.")
        print()
        print("  SOLUCION (2 minutos):")
        print("    1. Tecla Windows -> escribe 'proxy' -> 'Configuracion de proxy'")
        print("    2. Desactiva 'Usar un servidor proxy'")
        print("    3. Desconecta la VPN si tienes una activa")
        print("    4. Vuelve a ejecutar instalar.bat")
    elif codigo == ERR_PROXY_OTRO:
        print("  DIAGNOSTICO: hay un proxy configurado que no deja pasar.")
        if ini:
            print("  Esta escrito en el fichero de configuracion de pip que")
            print("  aparece arriba. Abrelo y borra o comenta esa linea 'proxy ='.")
        if proxy_persiste:
            print("  Ademas sigue puesto tras limpiar el entorno, asi que viene de")
            print("  la configuracion de Windows: Tecla Windows -> 'proxy' ->")
            print("  'Configuracion de proxy' -> desactiva 'Usar un servidor proxy'.")
        print("  Desconecta tambien la VPN si tienes una activa, y reintenta.")
    elif codigo == ERR_SIN_DNS:
        print("  DIAGNOSTICO: no se resuelven nombres de dominio. O no hay")
        print("  internet, o el DNS esta mal configurado.")
    elif codigo == ERR_TLS:
        print("  DIAGNOSTICO: fallo de certificados. Suele ser un antivirus que")
        print("  inspecciona el trafico HTTPS. Desactiva su escaneo SSL/HTTPS.")
    else:
        print("  DIAGNOSTICO: no hay salida a internet desde Python.")
        print("  Comprueba la conexion y que el firewall no bloquee a python.exe.")

    return codigo


if __name__ == "__main__":
    sys.exit(main())
