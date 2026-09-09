#!/bin/bash
set -euo pipefail

BACKUP_DIR="/backup"
PROJECT_DIR="/opt/zhihui"
COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
DATE=$(date +%Y%m%d)
HOUR=$(date +%H)

mkdir -p "$BACKUP_DIR"

case "${1:-}" in
  volumes)
    # 每日凌晨：停服备份持久化卷（排除 MongoDB，其走 mongodump 单独备份）
    cd "$PROJECT_DIR"
    echo "[$(date)] 停止 Milvus 准备备份..."
    docker compose $COMPOSE_FILES stop milvus
    tar -czf "$BACKUP_DIR/volumes-${DATE}.tar.gz" \
      --exclude='volumes/mongo' \
      volumes/
    echo "[$(date)] 重启 Milvus..."
    docker compose $COMPOSE_FILES start milvus
    echo "[$(date)] 卷备份完成: volumes-${DATE}.tar.gz"
    # 保留 14 天
    find "$BACKUP_DIR" -name 'volumes-*.tar.gz' -mtime +14 -delete
    ;;
  mongo)
    # 每 6 小时：在线备份 MongoDB
    cd "$PROJECT_DIR"
    ARCHIVE="$BACKUP_DIR/mongo-${DATE}-${HOUR}.archive"
    docker compose $COMPOSE_FILES exec -T mongo mongodump \
      --db kb001 --username zhihui_user --password "${MONGO_BACKUP_PASS:-}" \
      --authenticationDatabase kb001 --archive > "$ARCHIVE"
    echo "[$(date)] MongoDB 备份完成: mongo-${DATE}-${HOUR}.archive"
    # 保留 14 天
    find "$BACKUP_DIR" -name 'mongo-*.archive' -mtime +14 -delete
    ;;
  *)
    echo "Usage: $0 {volumes|mongo}"
    exit 1
    ;;
esac
