from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core import Workspace, Dataset, RunConfiguration, Experiment
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.runconfig import CondaDependencies
from azureml.train.dnn import PyTorch
from azureml.pipeline.steps import PythonScriptStep, EstimatorStep


def find_or_create_compute_target(workspace, name, vm_size='STANDARD_NC6', min_nodes=1, max_nodes=6, verbose=True):
    try:
        target = ComputeTarget(workspace=workspace, name=name)
    except ComputeTargetException:
        config = AmlCompute.provisioning_configuration(vm_size=vm_size, min_nodes=min_nodes, max_nodes=max_nodes)
        target = ComputeTarget.create(workspace, name, config)
        target.wait_for_completion(show_output=verbose)
    return target


if __name__ == '__main__':
    project_folder = 'glue'
    train_script_name = 'train.py'
    prepare_script_name = 'prepare.py'
    experiment_name = 'glue_experiment'
    compute_target_name = 'gpu-cluster'
    vm_size = 'STANDARD_NC12'  # 'STANDARD_D2_V2'
    data_dir = 'glue_data'
    prepared_data_dir = f"prepared_{data_dir}"
    # Change to false if you already have it up there
    # TODO - Super lazy, just uploading full folder. there are useless files in there.
    upload_data = True

    ######################################
    # AZURE BOILERPLATE
    ######################################
    workspace = Workspace.from_config()
    datastore = workspace.get_default_datastore()
    experiment = Experiment(workspace=workspace, name=experiment_name)
    compute_target = find_or_create_compute_target(
        workspace,
        compute_target_name,
        vm_size=vm_size,
        max_nodes=4
    )

    ######################################
    # Optionally upload data
    ######################################
    if upload_data:
        datastore.upload(
            src_dir=data_dir,
            target_path=data_dir,
            overwrite=True,
            show_progress=True
        )

    ######################################
    # Dataset configuration
    ######################################
    raw_dataset = Dataset.File.from_files([(datastore, data_dir)])
    prepared_dataset = PipelineData(prepared_data_dir, datastore=datastore).as_dataset()
    prepared_dataset = prepared_dataset.register(name=prepared_data_dir)

    ######################################
    # Pipeline environment configuration
    ######################################
    conda = CondaDependencies.create(
        pip_packages=['azureml-sdk', 'azureml-dataprep[fuse,pandas]', 'torch', 'transformers'],
        pin_sdk_version=False
    )
    conda.set_pip_option('--pre')
    run_config = RunConfiguration()
    run_config.environment.python.conda_dependencies = conda

    # ######################################
    # # Build estimator + pipeline
    # ######################################
    estimator = PyTorch(
        source_directory=project_folder,
        compute_target=compute_target,
        entry_script=train_script_name,
        use_gpu=True,
        pip_packages=['azureml-sdk', 'pytorch-lightning', 'transformers', 'scikit-learn'],
    )

    prepare_step = PythonScriptStep(
        name='prepare step',
        script_name=prepare_script_name,
        arguments=[],
        inputs=[raw_dataset.as_named_input(data_dir).as_mount()],
        outputs=[prepared_dataset],
        source_directory=project_folder,
        compute_target=compute_target,
        runconfig=run_config,
        allow_reuse=True
    )

    train_step = EstimatorStep(
        name='train step',
        estimator=estimator,
        estimator_entry_script_arguments=['--do_train', '--do_predict'],
        inputs=[prepared_dataset.as_mount()],
        compute_target=compute_target,
        allow_reuse=True
    )

    pipeline = Pipeline(workspace, steps=[prepare_step, train_step])
    run = experiment.submit(pipeline)
