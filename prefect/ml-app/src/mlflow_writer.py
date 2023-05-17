from mlflow.tracking import MlflowClient
from omegaconf import DictConfig, ListConfig, OmegaConf


class MlflowWriter:
    def __init__(self, experiment_name, **kwargs):
        self.client = MlflowClient(**kwargs)
        try:
            self.experiment_id = self.client.create_experiment(experiment_name)
        except:
            self.experiment_id = self.client.get_experiment_by_name(experiment_name).experiment_id

        self.run_id = self.client.create_run(self.experiment_id).info.run_id

        print(f"ExperimentID: {self.experiment_id}")
        print(f"RunID: {self.run_id}")

    def log_params_from_omegaconf_dict(self, params):
        if isinstance(params, dict):
            params = OmegaConf.create(params)
        for param_name, element in params.items():
            self._explore_recursive(param_name, element)

    def _explore_recursive(self, parent_name, element):
        if isinstance(element, DictConfig):
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                self._explore_recursive(f"{parent_name}.{k}", v)
            else:
                self.client.log_param(self.run_id, f"{parent_name}.{k}", v)
        elif isinstance(element, ListConfig):
            for i, v in enumerate(element):
                self.client.log_param(self.run_id, f"{parent_name}.{i}", v)
        else:
            self.client.log_param(self.run_id, parent_name, element)

    def log_param(self, key, value):
        self.client.log_param(self.run_id, key, value)

    def log_metric(self, key, value):
        self.client.log_metric(self.run_id, key, value)

    def log_artifact(self, local_path):
        self.client.log_artifact(self.run_id, local_path)

    def log_dict(self, dictionary, filename):
        self.client.log_dict(self.run_id, dictionary, filename)

    def log_figure(self, figure, filename):
        self.client.log_figure(self.run_id, figure, filename)

    def set_terminated(self):
        self.client.set_terminated(self.run_id)

    def set_tags(self, tags):
        for k, v in tags.items():
            self.client.set_tag(self.run_id, k, v)

    def create_new_run(self, tags=None):
        self.run = self.client.create_run(self.experiment_id, tags=tags)
        self.run_id = self.run.info.run_id
        print(f"New run started: {tags['mlflow.runName']}")
