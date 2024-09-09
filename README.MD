#  Two-Stage Segmentation of Pelvic Bone Fragments with Injuries in CT/X-ray Images Using nnUNet 

Here is the code for the solution from the SMILE team that ranked 2nd in Task1 and 1st in Task2 of the PENWIN Challenge.
(https://pengwin.grand-challenge.org/result/)

The models used in the two tasks were all trained using nnunetv2 with default configuration, 
the model weights can be download in https://drive.google.com/drive/folders/10vj-uYT8Gev8K0S0Skpq-rjvApdyTUbA?usp=drive_link


## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yuepeiyan/PENGWIN_Challenge.git
2. Install nnunetv2:
   ```bash
   cd nnUNet
   pip install -e .
3. Download the CT weights and place the `models` folder into the `Inference/CT` directory. (Using Task 1 as an example, the steps for Task 2 are identical.)

## Usage
1. Copy an image to the `test/input/images/plevic-fracture-ct` directory. 
2. Run `test_run.sh` to build and run a docker container that performs inference on the image placed in the folder mentioned above.
3. The output will be saved to the `test/output/images/plevic-fracture-ct-segmentation` directory. 

If you prefer not to run a Docker container, you can simply use the `inference_one_image` function in `two_stage_inference.py`. 
This function allows you to perform inference on a single image and save the output to a specified path.