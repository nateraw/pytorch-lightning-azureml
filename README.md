# Instructions

#### Make a new conda environment locally for development

```
conda create -n pl_azureml_env -y python=3.6 --no-default-packages
conda activate pl_azureml_env
pip install -r requirements.txt
```

#### Create a new AzureML resource in Azure Portal

![](images/create-workspace.gif)

#### From that resource, download the associated `config.json` file

![](images/download-config.png)

