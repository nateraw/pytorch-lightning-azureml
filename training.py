from azureml.core import Experiment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.train.dnn import PyTorch
from azureml.core.workspace import Workspace


def find_or_create_compute_target(workspace, name, vm_size='STANDARD_NC6', max_nodes=4, verbose=True):
    try:
        target = ComputeTarget(workspace=workspace, name=name)
    except ComputeTargetException:
        config = AmlCompute.provisioning_configuration(vm_size=vm_size, max_nodes=max_nodes)
        target = ComputeTarget.create(workspace, name, config)
        target.wait_for_completion(show_output=verbose)

    return target


if __name__ == '__main__':

    config_filepath = './config.json'
    project_folder = './mnist_example/'
    compute_cluster_name = 'gpu-cluster'
    experiment_name = 'mnist_experiment'

    # Init workspace object from azureml workspace resource you've created
    ws = Workspace.from_config(config_filepath)
    # Create or find a compute target in this workspace based on compute_cluster_name
    compute_target = find_or_create_compute_target(ws, compute_cluster_name)

    # Create/point to an experiment
    experiment = Experiment(ws, name=experiment_name)

    # Parameters passed to train.py
    script_params = {
        '--output_dir': './outputs',
        '--n_gpus': 1,
        '--learning_rate': 0.02
    }

    # Create Pytorch Estimator
    estimator = PyTorch(
        source_directory=project_folder,
        script_params=script_params,
        compute_target=compute_target,
        entry_script='train.py',
        use_gpu=True,
        pip_packages=['pytorch-lightning==0.7.1']
    )

    # Submit job
    run = experiment.submit(estimator)
