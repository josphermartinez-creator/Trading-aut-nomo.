"""El fichero .env tiene que llegar al proceso.

En Windows no hay systemd ni Docker: si nadie lee el .env, el usuario
rellena sus credenciales de MT5 y el bot arranca sin ellas, en silencio.
"""

from __future__ import annotations

import os

import pytest

from goldbot.config import load_dotenv


@pytest.fixture
def entorno_limpio(monkeypatch):
    for key in list(os.environ):
        if key.startswith("GOLDBOT_TEST_"):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_lee_pares_simples(tmp_path, entorno_limpio):
    env = tmp_path / ".env"
    env.write_text("GOLDBOT_TEST_LOGIN=123456\nGOLDBOT_TEST_SERVER=XMGlobal-MT5 2\n")

    applied = load_dotenv(env)

    assert applied["GOLDBOT_TEST_LOGIN"] == "123456"
    assert os.environ["GOLDBOT_TEST_SERVER"] == "XMGlobal-MT5 2"


def test_ignora_comentarios_y_lineas_vacias(tmp_path, entorno_limpio):
    env = tmp_path / ".env"
    env.write_text(
        "# comentario\n"
        "\n"
        "   \n"
        "GOLDBOT_TEST_A=1\n"
        "sin_signo_igual\n"
    )

    applied = load_dotenv(env)

    assert applied == {"GOLDBOT_TEST_A": "1"}


def test_quita_comillas_y_prefijo_export(tmp_path, entorno_limpio):
    env = tmp_path / ".env"
    env.write_text(
        'GOLDBOT_TEST_PASS="clave con espacios"\n'
        "export GOLDBOT_TEST_B='otra'\n"
    )

    load_dotenv(env)

    assert os.environ["GOLDBOT_TEST_PASS"] == "clave con espacios"
    assert os.environ["GOLDBOT_TEST_B"] == "otra"


def test_el_entorno_real_tiene_prioridad(tmp_path, entorno_limpio):
    entorno_limpio.setenv("GOLDBOT_TEST_MODE", "paper")
    env = tmp_path / ".env"
    env.write_text("GOLDBOT_TEST_MODE=mt5\n")

    applied = load_dotenv(env)

    assert os.environ["GOLDBOT_TEST_MODE"] == "paper"
    assert "GOLDBOT_TEST_MODE" not in applied


def test_la_contrasena_puede_llevar_signo_igual(tmp_path, entorno_limpio):
    env = tmp_path / ".env"
    env.write_text("GOLDBOT_TEST_PASS=ab=cd=ef\n")

    load_dotenv(env)

    assert os.environ["GOLDBOT_TEST_PASS"] == "ab=cd=ef"


def test_sin_fichero_no_falla(tmp_path):
    assert load_dotenv(tmp_path / "no-existe.env") == {}
