"""El proxy tiene que quedar neutralizado de la forma correcta.

Vaciar HTTP_PROXY y HTTPS_PROXY parece la solucion evidente y es justo lo
contrario: en Windows ``getproxies()`` es ``getproxies_environment() or
getproxies_registry()``, y ``getproxies_environment()`` descarta las
variables vacias. Dejarlas vacias hace que el diccionario quede vacio, que es
falsy, y entonces Python **consulta el proxy del registro de Windows** -- que
es donde lo dejan las VPN y los antivirus.

El invariante que hay que sostener: tras neutralizar, el diccionario de
entorno debe estar NO vacio (para que el registro no se consulte) pero sin
entrada para http ni https (para que la conexion sea directa).
"""

from __future__ import annotations

import urllib.request

import pytest

from scripts.diagnostico_red import (
    ERR_PROXY_OTRO,
    ERR_PROXY_SOCKS,
    ERR_SIN_CONEXION,
    ERR_SIN_DNS,
    OK,
    afinar_codigo,
    neutralizar_proxy,
    proxies_visibles,
)

VARIABLES = (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "FTP_PROXY", "PIP_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "pip_proxy",
    "NO_PROXY", "no_proxy",
)


@pytest.fixture
def entorno(monkeypatch):
    for nombre in VARIABLES:
        monkeypatch.delenv(nombre, raising=False)
    return monkeypatch


def test_el_entorno_no_queda_vacio(entorno):
    """Si quedara vacio, Windows caeria al proxy del registro."""
    entorno.setenv("HTTPS_PROXY", "socks5://127.0.0.1:1080")

    neutralizar_proxy()

    assert urllib.request.getproxies_environment(), (
        "el diccionario de entorno quedo vacio: en Windows eso hace que "
        "Python use el proxy del registro, que es el fallo que perseguimos"
    )


def test_no_queda_proxy_para_http_ni_https(entorno):
    entorno.setenv("HTTPS_PROXY", "socks5://127.0.0.1:1080")
    entorno.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")
    entorno.setenv("http_proxy", "http://127.0.0.1:8080")

    neutralizar_proxy()

    assert proxies_visibles() == {}


def test_neutralizar_es_idempotente(entorno):
    entorno.setenv("ALL_PROXY", "socks5://127.0.0.1:1080")

    neutralizar_proxy()
    primero = dict(urllib.request.getproxies_environment())
    neutralizar_proxy()

    assert urllib.request.getproxies_environment() == primero


def test_se_ignoran_las_variables_que_no_son_de_pip(entorno):
    """npm_config_https_proxy y companía aparecen en getproxies() y no importan."""
    entorno.setenv("npm_config_https_proxy", "http://127.0.0.1:9000")

    assert proxies_visibles() == {}


def test_un_socks_se_clasifica_como_socks(entorno, monkeypatch):
    """El mensaje de pip trae 'SOCKS'; hay que reconocerlo para saber que decir."""
    import scripts.diagnostico_red as diag

    monkeypatch.setattr(diag.socket, "getaddrinfo", lambda *a, **k: [()])

    def falla(*_a, **_k):
        raise urllib.error.URLError("Missing dependencies for SOCKS support")

    monkeypatch.setattr(diag.urllib.request, "urlopen", falla)

    codigo, _ = diag.probar_pypi()
    assert codigo == ERR_PROXY_SOCKS


def test_sin_dns_se_distingue_de_sin_proxy(entorno, monkeypatch):
    import socket

    import scripts.diagnostico_red as diag

    def sin_dns(*_a, **_k):
        raise socket.gaierror("Name or service not known")

    monkeypatch.setattr(diag.socket, "getaddrinfo", sin_dns)

    codigo, _ = diag.probar_pypi()
    assert codigo == ERR_SIN_DNS
    assert codigo != OK


def test_un_timeout_con_proxy_en_pie_se_culpa_al_proxy():
    """Un timeout pelado no dice SOCKS, pero si el proxy sobrevivio, es el."""
    assert afinar_codigo(ERR_SIN_CONEXION, True, None) == ERR_PROXY_OTRO
    assert afinar_codigo(ERR_SIN_CONEXION, False, "pip.ini -> proxy = x") == ERR_PROXY_OTRO


def test_un_timeout_sin_proxy_se_queda_como_sin_conexion():
    assert afinar_codigo(ERR_SIN_CONEXION, False, None) == ERR_SIN_CONEXION


def test_afinar_no_pisa_un_diagnostico_ya_concreto():
    """Si ya sabemos que es SOCKS o DNS, el consejo generico solo estorbaria."""
    assert afinar_codigo(ERR_PROXY_SOCKS, True, None) == ERR_PROXY_SOCKS
    assert afinar_codigo(ERR_SIN_DNS, True, None) == ERR_SIN_DNS
    assert afinar_codigo(OK, True, "pip.ini -> proxy = x") == OK
