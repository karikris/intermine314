#!/usr/bin/env bash

# Set up Solr for InterMine testmine. The setup script populates these cores.

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <workspace_dir>"
    exit 2
fi

WORKSPACE_DIR=$1

require_cmd() {
    local cmd=$1
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "Missing required command: $cmd" >&2
        exit 1
    fi
}

require_cmd curl
require_cmd jq
require_cmd tar

cd "$WORKSPACE_DIR"

SOLR_VERSION="${SOLR_VERSION:-8.6.2}"
SOLR_BASE_URL="${SOLR_BASE_URL:-https://archive.apache.org/dist/lucene/solr}"
SOLR_PACKAGE="solr-${SOLR_VERSION}.tgz"
SOLR_DIR="solr-${SOLR_VERSION}"
SOLR="${SOLR_DIR}/bin/solr"

create_solr_core() {
    local core_name=$1
    local status

    status="$(curl -fsS "http://localhost:8983/solr/admin/cores?action=STATUS&core=${core_name}" \
        | jq -r --arg core_name "${core_name}" '.status[$core_name] // "{}"')"

    if [ "$status" = "{}" ]; then
        "$SOLR" create -c "${core_name}"
    else
        echo "Solr core ${core_name} already exists"
    fi
}

if [ ! -d "$SOLR_DIR" ]; then
    if [ ! -f "$SOLR_PACKAGE" ]; then
        curl -fsSL "${SOLR_BASE_URL}/${SOLR_VERSION}/${SOLR_PACKAGE}" -o "$SOLR_PACKAGE"
    fi
    tar xzf "$SOLR_PACKAGE"
fi

"$SOLR" restart
create_solr_core intermine-search
create_solr_core intermine-autocomplete
