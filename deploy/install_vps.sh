#!/usr/bin/env bash
# =====================================================================
# GoldBot - instalacion en un VPS Ubuntu/Debian limpio
#
#   curl -fsSL <url>/install_vps.sh | sudo bash
#   o bien:  sudo bash deploy/install_vps.sh
# =====================================================================
set -euo pipefail

INSTALL_DIR="${INSTALL_DIR:-/opt/goldbot}"
REPO_URL="${REPO_URL:-https://github.com/josphermartinez-creator/Trading-aut-nomo.git}"
SERVICE_USER="goldbot"

log()  { echo -e "\033[32m[+]\033[0m $*"; }
warn() { echo -e "\033[33m[!]\033[0m $*"; }
die()  { echo -e "\033[31m[x]\033[0m $*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "Ejecuta este script como root (sudo)."

log "Actualizando paquetes del sistema..."
apt-get update -qq
apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip git curl build-essential tzdata

log "Fijando la zona horaria en UTC (todo el bot razona en UTC)..."
timedatectl set-timezone UTC || warn "No se pudo fijar la zona horaria"

if ! id "$SERVICE_USER" &>/dev/null; then
    log "Creando el usuario de servicio '$SERVICE_USER'..."
    useradd --system --create-home --shell /bin/bash "$SERVICE_USER"
fi

if [[ -d "$INSTALL_DIR/.git" ]]; then
    log "Actualizando la instalacion existente en $INSTALL_DIR..."
    git -C "$INSTALL_DIR" pull --ff-only
else
    log "Clonando el repositorio en $INSTALL_DIR..."
    git clone "$REPO_URL" "$INSTALL_DIR"
fi

log "Creando el entorno virtual..."
python3 -m venv "$INSTALL_DIR/.venv"
"$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install --quiet -r "$INSTALL_DIR/requirements.txt"
"$INSTALL_DIR/.venv/bin/pip" install --quiet -e "$INSTALL_DIR"

mkdir -p "$INSTALL_DIR"/{data,logs,artifacts}

if [[ ! -f "$INSTALL_DIR/.env" ]]; then
    log "Creando .env a partir de la plantilla..."
    cp "$INSTALL_DIR/.env.example" "$INSTALL_DIR/.env"
    chmod 600 "$INSTALL_DIR/.env"
    warn "Edita $INSTALL_DIR/.env con tus credenciales antes de operar en real."
fi

chown -R "$SERVICE_USER:$SERVICE_USER" "$INSTALL_DIR"

log "Instalando el servicio systemd..."
cp "$INSTALL_DIR/deploy/goldbot.service" /etc/systemd/system/
systemctl daemon-reload
systemctl enable goldbot

cat <<EOF

======================================================================
 Instalacion completada en $INSTALL_DIR
======================================================================

 SIGUIENTE PASO OBLIGATORIO: descubrir estrategias antes de operar.

   sudo -u $SERVICE_USER $INSTALL_DIR/.venv/bin/goldbot learn --bootstrap

 Esto puede tardar entre 30 y 120 minutos: descarga el historico, hace
 evolucionar miles de estrategias y valida las supervivientes.

 Despues, arranca el servicio:

   sudo systemctl start goldbot
   sudo journalctl -u goldbot -f

 Comprobar el estado:

   sudo -u $SERVICE_USER $INSTALL_DIR/.venv/bin/goldbot status

 IMPORTANTE: el bot arranca con dry_run=true y NO enviara ordenes
 reales. Dejalo incubando varias semanas y revisa los resultados antes
 de plantearte cambiarlo.

======================================================================
EOF
