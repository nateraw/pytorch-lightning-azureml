from azureml.core import Dataset, Experiment, RunConfiguration, Workspace
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import CondaDependencies, MpiConfiguration
from azureml.pipeline.core import Pipeline, PipelineData, PipelineParameter, StepSequence
from azureml.pipeline.steps import EstimatorStep, PythonScriptStep
from azureml.train.dnn import PyTorch


def find_or_create_compute_target(
    workspace,
    name,
    vm_size="STANDARD_NC6",
    min_nodes=0,
    max_nodes=1,
    idle_seconds_before_scaledown=1200,
    vm_priority="lowpriority",
):
    try:
        target = ComputeTarget(workspace=workspace, name=name)
    except ComputeTargetException:
        config = AmlCompute.provisioning_configuration(
            vm_size=vm_size, min_nodes=min_nodes, max_nodes=max_nodes, vm_priority=vm_priority
        )
        target = ComputeTarget.create(workspace, name, config)
        target.wait_for_completion(show_output=True)
    return target


# The relative path to the folder that holds our scripts
prep_project_folder = "glue_prep"
train_project_folder = "glue_train"

# Name of script to run data preprocessing/preparation
prepare_script_name = "prepare.py"

# Name of script that will do training
train_script_name = "train.py"

# Your runs will get grouped within this experiment name (name it whatever ya want)
experiment_name = "glue_benchmark_mprc"

# A name for your compute target (name it whatever ya want)
compute_target_name = "gpu-cluster"

# Azure specific VM name. This one has a K80 GPU, 6 cores, 56GB RAM, and 380GB Disk Space
# Here's a good link on different VMs and their pricing...
# https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/
vm_size = "STANDARD_NV24"  # "STANDARD_NV12"

# The reference name to the path of your prepared/preprocessed data
# This will come through as an env variable in prepare.py
prepared_data_dir = "prepared_data"

workspace = Workspace.from_config()
datastore = workspace.get_default_datastore()
experiment = Experiment(workspace=workspace, name=experiment_name)
compute_target = find_or_create_compute_target(workspace, compute_target_name, vm_size=vm_size)

prepared_dataset = PipelineData(prepared_data_dir, datastore=datastore).as_dataset()
prepared_dataset = prepared_dataset.register(name=prepared_data_dir)

conda = CondaDependencies.create(
    pip_packages=[
        "azureml-sdk",
        "azureml-dataprep[fuse,pandas]",
        "torch==1.5.0",
        "nlp==0.2.0",
        "transformers==2.11.0",
    ],
    pin_sdk_version=False,
)
conda.set_pip_option("--pre")
run_config = RunConfiguration()
run_config.environment.python.conda_dependencies = conda

# Define Pipeline Parameters
model_name_param = PipelineParameter("model_name_or_path", "bert-base-cased")
max_seq_len_param = PipelineParameter("max_seq_length", 128)
task_param = PipelineParameter("task", "mrpc")
learning_rate_param = PipelineParameter("learning_rate", 2e-5)
seed_param = PipelineParameter("seed", 1)
train_batch_size_param = PipelineParameter("train_batch_size", 64)
eval_batch_size_param = PipelineParameter("eval_batch_size", 64)
max_epochs_param = PipelineParameter("max_epochs", 3)
num_gpus_param = PipelineParameter("gpus", 2)
num_workers_param = PipelineParameter("num_workers", 2)


prepare_step = PythonScriptStep(
    name="Preparation Step",
    script_name=prepare_script_name,
    arguments=["--model_name_or_path", model_name_param, "--max_seq_length", max_seq_len_param],
    outputs=[prepared_dataset],
    source_directory=prep_project_folder,
    compute_target=compute_target,
    runconfig=run_config,
    allow_reuse=True,
)

estimator = PyTorch(
    source_directory=train_project_folder,
    compute_target=compute_target,
    entry_script=train_script_name,
    use_gpu=True,
    pip_packages=[
        "azureml-sdk",
        "nlp==0.2.0",
        "pytorch-lightning==0.8.0rc4",
        "transformers==2.11.0",
        "pandas",
        "scipy",
        "scikit-learn",
    ],
    framework_version="1.5",
)

train_step = EstimatorStep(
    name="Training Step",
    estimator=estimator,
    estimator_entry_script_arguments=[
        "--model_name_or_path",
        model_name_param,
        "--task",
        task_param,
        "--max_seq_length",
        max_seq_len_param,
        "--max_epochs",
        max_epochs_param,
        "--learning_rate",
        learning_rate_param,
        "--seed",
        seed_param,
        "--gpus",
        num_gpus_param,
        "--num_workers",
        num_workers_param,
        "--train_batch_size",
        train_batch_size_param,
        "--eval_batch_size",
        eval_batch_size_param,
        "--output_dir",
        "./outputs",
        "--do_train",
        "--do_predict",
    ],
    inputs=[prepared_dataset.as_mount()],
    compute_target=compute_target,
)

step_sequence = StepSequence(steps=[prepare_step, train_step])
pipeline = Pipeline(workspace, steps=step_sequence)

# Submit single experiment run
run = experiment.submit(pipeline)

# Run the three listed models over 5 random seeds. (15 experiment runs total)
# for seed in range(5):
#     for model in ["distilbert-base-cased", "bert-base-cased", "albert-base-v2"]:
#         run = experiment.submit(
#             pipeline,
#             pipeline_parameters={
#                 "model_name_or_path": model,
#                 "task": "cola",
#                 "train_batch_size": 32,
#                 "eval_batch_size": 32,
#                 "gpus": 4,
#                 "seed": seed,
#                 "num_workers": 16,
#                 "max_epochs": 4,
#             },
#         )
