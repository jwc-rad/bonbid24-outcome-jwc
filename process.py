"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./export.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""
from pathlib import Path
import json
import os
from glob import glob
import SimpleITK
import numpy as np
import torch

import torch.nn as nn

### algorithm
from src.test_cls import run_inference

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")


def get_default_device():
    ######## set device#########
    if torch.cuda.is_available():
        print ("Using gpu device")
        return torch.device('cuda')
    else:
        print ("Using cpu device")
        return torch.device('cpu')


class BaseNet(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self,x):
        out=torch.where(x<-2,1,0)
        return out

def run():
    # Read the input
    """
    input_skull_stripped_adc = load_image_file_as_array(
        location=INPUT_PATH / "images/skull-stripped-adc-brain-mri",
    )
    z_adc = load_image_file_as_array(
        location=INPUT_PATH / "images/z-score-adc",
    )
    """
    
    # Read the input
    input_skull_stripped_adc = load_image_file_as_sitkimage(
        location=INPUT_PATH / "images/skull-stripped-adc-brain-mri",
    )
    z_adc = load_image_file_as_sitkimage(
        location=INPUT_PATH / "images/z-score-adc",
    )
    
    # Process the inputs: any way you'd like
    ens_preds_bin, list_preds_bin, list_preds, list_thrs = run_inference(
        [input_skull_stripped_adc, z_adc],
        [
            '2024-10-10_04-07-11', # 64/128, effB0, dropout 0.5
            '2024-11-02_11-18-26', # 96/160, res18, dropout 0.5
            #'2024-10-07_02-55-10', # 96/160, effB0, dropout 0.2
            #'2024-10-09_03-09-58', # 64/128, effB0, dropout 0.5
        ],
    )
    
    print(f"list_preds: {list_preds}")
    print(f"thresholds: {list_thrs}")
    print(f"list_preds_bin: {list_preds_bin}")
    print(f"ensemble preds: {ens_preds_bin}")
    
    #hie_segmentation=SimpleITK.GetImageFromArray(out)    
    # 
    output_2_year_neurocognitive_outcome = ens_preds_bin

    # Save your output
    write_json_file(
        location=OUTPUT_PATH / "2-year-neurocognitive-outcome.json",
        content=output_2_year_neurocognitive_outcome
    )
    
    return 0


def write_json_file(*, location, content):
    # Writes a json file
    with open(location, 'w') as f:
        f.write(json.dumps(content, indent=4))



def save_image(*, pred_lesions):
    relative_path="images/hie-lesion-segmentation"
    output_directory = OUTPUT_PATH / relative_path

    output_directory.mkdir(exist_ok=True, parents=True)

    file_save_name=output_directory / "overlay.mha"
    print (file_save_name)

    SimpleITK.WriteImage(pred_lesions, file_save_name)
    check_file = os.path.isfile(file_save_name)
    print ("check file", check_file)



def load_image_file_as_array(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    # Convert it to a Numpy array
    return SimpleITK.GetArrayFromImage(result)

def load_image_file_as_sitkimage(*, location):
    # Use SimpleITK to read a file
    input_files = glob(str(location / "*.mha"))
    result = SimpleITK.ReadImage(input_files[0])

    return result

def _show_torch_cuda_info():


    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())