pipenv lock --requirements --keep-outdated > api/requirements.txt
docker build -t aks_experiment_runner -f training/Dockerfile .