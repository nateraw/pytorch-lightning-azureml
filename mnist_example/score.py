import json
import os

import torch
import torch.nn.functional as F

from mnist_model import MNISTModel

from azureml.core.model import Model


def init():
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    # For multiple models, it points to the folder containing all deployed models (./azureml-models)
    ckpt_filename = os.listdir(os.getenv('AZUREML_MODEL_DIR'))[0]
    ckpt_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), ckpt_filename)
    model = MNISTModel.load_from_checkpoint(ckpt_path)
    model.eval()
    model.freeze()


def run(input_data):
    input_data = torch.tensor(json.loads(input_data)['data'])

    classes = [
        'zero',
        'one',
        'two',
        'three',
        'four',
        'five',
        'six',
        'seven',
        'eight',
        'nine'
    ]

    output = model(input_data)
    pred_probs = F.softmax(output, dim=1)[0]
    index = torch.argmax(pred_probs, dim=0)

    result = {
        "label": classes[index].title(),
        "probability": str(pred_probs[index].item())
    }

    return result
