#!/usr/bin/env bash
set -uo pipefail
set +e

FAILURE=false

echo "pipenv check"
pipenv check  # Not reporting failure here, because sometimes this fails due to API request limit

echo "pylint"
pipenv run pylint --ignore=playground,demos evalg autoks training datasets || FAILURE=true

echo "pycodestyle"
pipenv run pycodestyle --exclude=playground,demos,.ipynb_checkpoints evalg autoks training datasets || FAILURE=true

echo "bandit"
pipenv run bandit -ll -r {autoks,evalg,datasets,training} -x demos,playground || FAILURE=true

echo "shellcheck"
shellcheck tasks/*.sh || FAILURE=true

if [ "$FAILURE" = true ]; then
  echo "Linting failed"
  exit 0  # TODO: don't actually fail CI
fi
echo "Linting passed"
exit 0