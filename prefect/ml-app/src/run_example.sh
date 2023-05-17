#!/bin/bash
cd `dirname $0`
cd ../env/

: 'hydra only'
# shot
docker-compose run --rm python_dev_experiment \
  python3 src/train_hydra.py


# grid search
docker-compose run --rm python_dev_experiment \
  python3 src/train_hydra.py -m \
  'model.search_params.learning_rate=choice(0.01,0.1)' \
  'model.search_params.max_depth=choice(3,6)'


# bayesian search(configのsweeper uncommentoutする)
docker-compose run --rm python_dev_experiment \
  python3 src/train_hydra.py -m \





: 'hydra+mlflow only'
# shot
docker-compose run --rm python_dev_experiment \
  python3 src/train.py


# grid search
docker-compose run --rm python_dev_experiment \
  python3 src/train.py -m \
  'model.train_params.learning_rate=choice(0.01,0.1)' \
  'model.train_params.max_depth=choice(3,6)'


# bayesian search(configのsweeper uncommentoutする)
docker-compose run --rm python_dev_experiment \
  python3 src/train.py -m
