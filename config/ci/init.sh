#!/bin/bash

set -euo pipefail

if [ "$#" != "2" ]; then
   echo "Usage: $0 <workspace_dir> <testmodel_url>"
   exit 2
fi

WORKSPACE_DIR=$1
TESTMODEL_URL=$2

INTERMINE_DIR=${WORKSPACE_DIR}/server
TESTMINE_DIR=${INTERMINE_DIR}/testmine

cd "${WORKSPACE_DIR}"

# Pull in the server code.

if [ ! -d "${INTERMINE_DIR}" ]; then
    git clone --single-branch --depth 1 -b dev https://github.com/ucam-department-of-psychiatry/intermine.git "${INTERMINE_DIR}"
fi

export PSQL_USER=test
export PSQL_PWD=test
export PSQL_HOST=localhost
export PGPASSWORD=${PGPASSWORD:-postgres}
export KEYSTORE=${PWD}/keystore.jks

echo "#---> Running unit tests"

sudo -E -u postgres dropdb -h "$PSQL_HOST" --if-exists intermine-demo
sudo -E -u postgres dropdb -h "$PSQL_HOST" --if-exists userprofile-demo

sudo -E -u postgres dropuser -h "${PSQL_HOST}" --if-exists test
sudo -E -u postgres createuser -h "${PSQL_HOST}" test
sudo -E -u postgres psql -h "${PSQL_HOST}" -c "alter user test with encrypted password 'test';"

# Set up properties
PROPDIR=${HOME}/.intermine
TESTMODEL_PROPS=${PROPDIR}/testmodel.properties

mkdir -p "${PROPDIR}"

echo "#--- creating ${TESTMODEL_PROPS}"
cp "${INTERMINE_DIR}"/config/testmodel.properties "${TESTMODEL_PROPS}"
sed -i -e "s/PSQL_HOST/${PSQL_HOST}/" "$TESTMODEL_PROPS"
sed -i -e "s/PSQL_USER/${PSQL_USER}/" "$TESTMODEL_PROPS"
sed -i -e "s/PSQL_PWD/${PSQL_PWD}/" "$TESTMODEL_PROPS"


# We will need a fully operational web-application
echo '#---> Building and releasing web application to test against'
(cd "${TESTMINE_DIR}" && ./setup.sh "${INTERMINE_DIR}")
# Travis is so slow
sleep 90 # let webapp startup

# Warm up the keyword search by requesting results, but ignoring the results
wget -O - "${TESTMODEL_URL}/service/search" > /dev/null
# Start any list upgrades
wget -O - "${TESTMODEL_URL}/service/lists?token=test-user-token" > /dev/null
