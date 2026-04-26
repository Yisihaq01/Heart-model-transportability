#!/bin/sh
set -e

if [ ! -f "data/ingestion_report.json" ]; then
  echo "ingestion_report.json not found. Running ingestion..."
  python -m src.ingest
fi

exec "$@"
