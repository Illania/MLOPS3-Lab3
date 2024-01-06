#!/bin/bash

set -e

# drift minio buckets
docker compose up -d --build minio
sleep 10
docker compose exec minio sh -c 'mc mb -p local/$MLFLOW_BUCKET_NAME'
docker compose exec minio sh -c 'mc anonymous set public local/$MLFLOW_BUCKET_NAME'
docker compose exec minio sh -c 'mc admin user svcacct add local $MINIO_ROOT_USER --access-key "$MINIO_ACCESS_KEY" --secret-key "$MINIO_SECRET_ACCESS_KEY"'
