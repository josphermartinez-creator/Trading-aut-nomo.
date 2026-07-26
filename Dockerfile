# =====================================================================
# GoldBot - imagen de produccion
# =====================================================================
FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=UTC

# Dependencias del sistema. build-essential se necesita para compilar
# algunas ruedas de scipy/pyarrow en arquitecturas sin binarios.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Las dependencias se copian primero para aprovechar la cache de capas:
# el codigo cambia a diario, las dependencias casi nunca.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY goldbot/ ./goldbot/
COPY configs/ ./configs/
COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir -e .

# Usuario sin privilegios: un bot que ejecuta codigo evolucionado no
# deberia correr como root.
RUN useradd --create-home --shell /bin/bash goldbot \
    && mkdir -p /app/data /app/logs /app/artifacts \
    && chown -R goldbot:goldbot /app
USER goldbot

# Volumenes para que el historico, la base de datos y los modelos
# sobrevivan a la recreacion del contenedor.
VOLUME ["/app/data", "/app/logs", "/app/artifacts"]

HEALTHCHECK --interval=5m --timeout=30s --start-period=2m --retries=3 \
    CMD python -c "import sys; from goldbot.config import load_config; load_config(); sys.exit(0)" || exit 1

ENTRYPOINT ["goldbot"]
CMD ["schedule"]
