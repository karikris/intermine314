#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <workspace_dir> <testmodel_url>"
    exit 2
fi

WORKSPACE_DIR=$1
TESTMODEL_URL=$2
SERVICE_URL="${TESTMODEL_URL%/}"
case "${SERVICE_URL}" in
    */service) ;;
    *) SERVICE_URL="${SERVICE_URL}/service" ;;
esac

INTERMINE_REPO="${INTERMINE_REPO:-https://github.com/intermine/intermine.git}"
INTERMINE_REF="${INTERMINE_REF:-dev}"
INTERMINE_DIR="${WORKSPACE_DIR}/server"
TESTMINE_DIR="${INTERMINE_DIR}/testmine"

require_cmd() {
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

require_cmd git
require_cmd sed
require_cmd curl
require_cmd sudo

cd "${WORKSPACE_DIR}"

if [ ! -d "${INTERMINE_DIR}" ]; then
    git clone --single-branch --depth 1 -b "${INTERMINE_REF}" "${INTERMINE_REPO}" "${INTERMINE_DIR}"
fi

export PSQL_USER="${PSQL_USER:-test}"
export PSQL_PWD="${PSQL_PWD:-test}"
export PSQL_HOST="${PSQL_HOST:-localhost}"
export PGPASSWORD="${PGPASSWORD:-postgres}"
export KEYSTORE="${KEYSTORE:-${PWD}/keystore.jks}"

echo "#---> Resetting postgres test users and databases"
sudo -E -u postgres dropdb -h "${PSQL_HOST}" --if-exists intermine-demo
sudo -E -u postgres dropdb -h "${PSQL_HOST}" --if-exists userprofile-demo
sudo -E -u postgres dropuser -h "${PSQL_HOST}" --if-exists "${PSQL_USER}"
sudo -E -u postgres createuser -h "${PSQL_HOST}" "${PSQL_USER}"
sudo -E -u postgres psql -h "${PSQL_HOST}" -c "alter user ${PSQL_USER} with encrypted password '${PSQL_PWD}';"

PROPDIR="${HOME}/.intermine"
TESTMODEL_PROPS="${PROPDIR}/testmodel.properties"
mkdir -p "${PROPDIR}"

echo "#---> Creating ${TESTMODEL_PROPS}"
cp "${INTERMINE_DIR}/config/testmodel.properties" "${TESTMODEL_PROPS}"
sed -i -e "s/PSQL_HOST/${PSQL_HOST}/" "${TESTMODEL_PROPS}"
sed -i -e "s/PSQL_USER/${PSQL_USER}/" "${TESTMODEL_PROPS}"
sed -i -e "s/PSQL_PWD/${PSQL_PWD}/" "${TESTMODEL_PROPS}"

echo "#---> Building and releasing InterMine test web application"
(cd "${TESTMINE_DIR}" && ./setup.sh "${INTERMINE_DIR}")

echo "#---> Waiting for webservice startup"
for _ in $(seq 1 60); do
    if curl -fsS "${SERVICE_URL}/version/ws" >/dev/null; then
        break
    fi
    sleep 5
done

if ! curl -fsS "${SERVICE_URL}/version/ws" >/dev/null; then
    echo "Timed out waiting for ${SERVICE_URL}" >&2
    exit 1
fi

echo "#---> Warming service endpoints"
curl -fsS "${SERVICE_URL}/search" >/dev/null
curl -fsS "${SERVICE_URL}/lists?token=test-user-token" >/dev/null
