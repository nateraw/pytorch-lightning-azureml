import numpy as np

from azureml.core import Experiment
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from azureml.core.workspace import Workspace

# InferenceConfig(
#     entry_script,
#     runtime=None,
#     conda_file=None,
#     xtra_docker_file_steps=None,
#     source_directory=None,
#     enable_gpu=None,
#     description=None,
#     base_image=None,
#     base_image_registry=None,
#     cuda_version=None,
#     environment=None
# )

# Environment.from_pip_requirements(name, file_path)


def get_best_experiment_run(experiment, metric='best_val_loss', mode='min'):
    # if we can't find the specified metric in a given run, replace with infinity
    fill_value = np.inf if mode == 'min' else -np.inf

    # Get a list of all run objects from specified experiment
    runs = list(experiment.get_runs())

    # Get list of all values for specified metric for each run
    all_run_metrics = []
    for run in runs:
        metrics = run.get_metrics()
        run_metric = metrics.get(metric, fill_value)
        all_run_metrics.append(run_metric)

    # Get the idx of the best run
    if mode == 'min':
        best_run_idx = np.argmin(all_run_metrics)
    else:
        best_run_idx = np.argmax(all_run_metrics)

    return runs[best_run_idx]


if __name__ == '__main__':
    config_filepath = './config.json'
    project_folder = './mnist_example/'
    compute_cluster_name = 'gpu-cluster'
    experiment_name = 'mnist_experiment'
    model_name = 'mnist-model'
    service_name = 'mnist-service'

    # Init workspace object from azureml workspace resource you've created
    ws = Workspace.from_config(config_filepath)

    # Point to an experiment
    experiment = Experiment(ws, name=experiment_name)

    # Find the experiment's best run
    best_run = get_best_experiment_run(experiment)
    model_path = [f for f in best_run.get_file_names() if f.startswith('outputs') and f.endswith('.ckpt')][0]

    # Register your best run's model
    model = best_run.register_model(
        model_name=model_name,
        model_path=model_path
    )

    # Create an environment based on requirements
    environment = Environment.from_pip_requirements(
        'mnist_env',
        './inference_requirements.txt'
    )

    # Get inference config to configure API's behavior
    inference_config = InferenceConfig(
        source_directory=project_folder,
        entry_script="score.py",
        environment=environment,
        enable_gpu=False,
    )

    # Get deployment config which outlines your API service's specs
    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores=1,
        memory_gb=1,
        tags={'data': 'Handwritten Digits', 'method': 'Simple NN', 'framework': 'pytorch'},
        description='Classify handwritten MNIST digits using Pytorch Lightning'
    )

    # Deploy the service
    service = Model.deploy(
        workspace=ws,
        name=service_name,
        models=[model],
        inference_config=inference_config,
        deployment_config=deployment_config
    )

    service.wait_for_deployment(True)
