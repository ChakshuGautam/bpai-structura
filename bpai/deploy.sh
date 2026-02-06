#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────
REMOTE_HOST="azureuser@20.57.128.78"
REMOTE_DIR="/opt/saral"
DOMAIN="flywheel-dev.bioprocess.ai"

# ── 1. Install Docker on remote if missing ─────────────────────────────
echo "==> Checking Docker on remote..."
ssh "$REMOTE_HOST" 'bash -s' <<'INSTALL_DOCKER'
set -euo pipefail
if ! command -v docker &>/dev/null; then
    echo "Installing Docker..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
      https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
      sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    sudo usermod -aG docker "$USER"
    echo "Docker installed successfully."
else
    echo "Docker already installed: $(docker --version)"
fi
INSTALL_DOCKER

# ── 2. Rsync project files ─────────────────────────────────────────────
echo "==> Syncing project files to $REMOTE_HOST:$REMOTE_DIR ..."
ssh "$REMOTE_HOST" "sudo mkdir -p $REMOTE_DIR && sudo chown \$USER:\$USER $REMOTE_DIR"

rsync -avz --delete \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude 'node_modules' \
    --exclude '.env' \
    --exclude 'www/' \
    --exclude 'old-lib/' \
    --exclude 'structura-lib-dev/' \
    --exclude 'images/' \
    --exclude '*.log' \
    --exclude '*.pid' \
    --exclude '.knowledge/' \
    --exclude 'ocr-review-deploy/' \
    --exclude 'supabase/' \
    "$(dirname "$0")/../" "$REMOTE_HOST:$REMOTE_DIR/"

echo "==> Files synced."

# ── 3. Build and start services ────────────────────────────────────────
echo "==> Starting Docker Compose..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR/bpai && docker compose up -d --build"

# ── 4. Create MinIO buckets ────────────────────────────────────────────
echo "==> Creating MinIO buckets..."
ssh "$REMOTE_HOST" 'bash -s' <<'CREATE_BUCKETS'
set -euo pipefail
# Wait for MinIO to be healthy
for i in $(seq 1 30); do
    if docker exec $(docker ps -qf "name=minio") mc ready local 2>/dev/null; then
        break
    fi
    echo "  Waiting for MinIO... ($i/30)"
    sleep 2
done

# Configure mc alias inside the MinIO container
MINIO_CONTAINER=$(docker ps -qf "name=minio")
docker exec "$MINIO_CONTAINER" mc alias set local http://localhost:9000 minioadmin minioadmin 2>/dev/null || true

# Create buckets (ignore if they already exist)
docker exec "$MINIO_CONTAINER" mc mb --ignore-existing local/marker-pdf
docker exec "$MINIO_CONTAINER" mc mb --ignore-existing local/pdf-results
echo "MinIO buckets created: marker-pdf, pdf-results"
CREATE_BUCKETS

# ── 5. Verify services ─────────────────────────────────────────────────
echo "==> Checking service health..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR/bpai && docker compose ps"

# ── 6. SSL setup with certbot ──────────────────────────────────────────
echo ""
echo "==> To set up SSL, run the following on the remote host:"
echo ""
echo "    ssh $REMOTE_HOST"
echo "    sudo apt-get install -y certbot"
echo "    sudo certbot certonly --webroot -w /var/lib/docker/volumes/bpai_certbot_www/_data -d $DOMAIN"
echo "    cd $REMOTE_DIR/bpai && docker compose restart nginx"
echo ""
echo "==> Deployment complete!"
echo "    API (HTTP):  http://$DOMAIN/docs"
echo "    MinIO console: http://20.57.128.78:9001"
