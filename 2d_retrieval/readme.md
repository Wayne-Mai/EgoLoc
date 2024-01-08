# Video 2D Object Retrieval
## TODO

- [x] Init Code, Oct 2023
- [todo] Merge 2D retrieval code for validation and test, Jan 2024
- [] Add code for test set also, Jan 2024



This module is mostly adapted from Ego4D's VQ2D [official code](https://github.com/EGO4D/episodic-memory).

We remove the tracker of the pretrained model and then use the detector to select those peak predictions.

## Start from our precomputed data
The `2d_retrieval` part aims to get detected 2D bbox (`vq2d_val(test)_vq3d_peak_clip.json`, `baseline_vq3d_detection_v105_val(test).json`) and clips frames (`data/clips_frames/`) for given visual queries among the videos.

<!-- We provide our pre-computer 2D results in `data/precomputed/vq2d_val_vq3d_peak_only.json`. -->
We provide all the extracted clip frames and 2D detection results in [this shared Google Drive folder](https://drive.google.com/drive/folders/1hmCfNT3rolj_llIRc8UUKpaw1Chppv-Y?usp=sharing). So you don't have to follow all the complicated steps below.


## Start from scratch
Of course, you can use your own 2D prediction result to replace ours and don't forget to follow the same format.
In case you want to reproduce the 2D results by yourself, please kindly follow the instructions below:


## Environment
Set up the environment as [Ego4D VQ2D](https://github.com/EGO4D/episodic-memory) requested:
(If you encounter any trouble during VQ2D environment setup, please refer to [Ego4D VQ2D](https://github.com/EGO4D/episodic-memory)'s readme and issues.)


1. Clone the repository from [here](https://github.com/EGO4D/episodic-memory).

```bash
git clone git@github.com:EGO4D/episodic-memory.git
cd episodic-memory/VQ2D
export VQ2D_ROOT=$PWD
```
2. Create conda environment. [Ego4D VQ2D](https://github.com/EGO4D/episodic-memory) is using a slightly different Python environment.
```bash
conda create -n ego4d_vq2d python=3.8
```

3. Install [pytorch](https://pytorch.org/) using conda. We rely on `cuda-10.2` and `cudnn-7.6.5.32` for our experiments.
```bash
module load cuda/10.2.89
module load cudnn/7.6.5.32
conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

4. Install additional requirements using `pip`.
```bash
pip install -r requirements.txt

pip install -r requirements_new.txt
```

5. Install [detectron2](https://github.com/facebookresearch/detectron2).

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

python -m pip install detectron2 -f \
https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
```

6.  Install pytracking according to [these instructions](https://github.com/visionml/pytracking/blob/master/INSTALL.md). Download the pre-trained [KYS tracker weights](https://drive.google.com/drive/folders/1WGNcats9lpQpGjAmq0s0UwO6n22fxvKi) to `$VQ2D_ROOT/pretrained_models/kys.pth`.
```bash
cd $VQ2D_ROOT/dependencies
git clone git@github.com:visionml/pytracking.git
git checkout de9cb9bb4f8cad98604fe4b51383a1e66f1c45c0
```

Note: For installing the [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension) dependency for pytracking, follow these steps if the pip install fails.
```bash
cd $VQ2D_ROOT/dependencies
git clone git@github.com:ClementPinard/Pytorch-Correlation-extension.git
cd Pytorch-Correlation-extension
python setup.py install
```


## Data

For VQ3D task, you don't need to download VQ2D full videos but just the pretrained checkpoint and video clips for VQ3D split. The pretrained model can be downloaded using the ego4d CLI as follows:

```bash
python -m ego4d.cli.cli -y --output_directory data/pretrained_models --datasets vq2d_models
```
Note you can check their official page to see if there's a newer model available and the corresponding download command.
Then you need to follow Ego4D's [official code](https://github.com/EGO4D/episodic-memory) to download VQ3D video clips in `$VQ2D_ROOT` folder and prepare Ego4D's VQ2D code in `$VQ2D_ROOT` folder. Downloading the required video clips and extracting them into frames can take several days.

If you encounter problems with the [official code](https://github.com/EGO4D/episodic-memory) to process the videos to get video clips and frames, you can check our modified version for reference, in which we optimize the extraction to include VQ3D clips only to speed up the preprocessing. However, you may need to change the arguments according to your data structure.
```bash
# our modified video, clip and frame extractor
# if on a slurm server:
sbatch extract_clips.sh
sbatch extract_frames.sh

# else extract them locally:
python extract_clips_from_video.py
python extract_frames_from_clips_vq3d.py
```

We recommend to organize the file structure as follows:

```
- ego_loc
    - 2d_retrieval
        - [vq2d](Ego4D's VQ2D baseline code)
            - pretrained_models
        - [data](data for vq2d)
            - pretrained_models
            - [video_clips](only the video clips for VQ3D are need)
            - eg., vq_val.json, and other annotation json file from ego4d
            - eg., baseline_vq3d_detection_v105_val.json, extracted detection score for vq2d/vq3d val and test set
    - vq3d
    - data
```

## VQ2D Results

## Start from scratch

1. Download the annotations and videos as instructed [here](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md) to `$VQ2D_ROOT/data`.
```bash
python -m ego4d.cli.cli --output_directory="$VQ2D_ROOT/data" --datasets full_scale annotations
# Define ego4d videos directory
export EGO4D_VIDEOS_DIR=$VQ2D_ROOT/data/v1/full_scale
# Move out vq annotations to $VQ2D_ROOT/data
mv $VQ2D_ROOT/data/v1/annotations/vq_*.json $VQ2D_ROOT/data


# our path
EGO4D_ROOT=/ibex/ai/reference/CV/Ego4D/ego4d_data/v1/
cp $EGO4D_ROOT/annotations/vq_*.json data/
```

2. Process the VQ dataset.
```bash
python process_vq_dataset.py --annot-root data --save-root data
```

3. You don't have to train the VQ2D model since it takes around two weeks on an eight-GPU server. But you can refer to Ego4D's VQ2D [official code](https://github.com/EGO4D/episodic-memory) for detailed training instructions.

4. We download pretrained checkpoints by Ego3D as illustrated in [Data section](##Data). Since EgoLoc doesn't need the tracking result, you can skip step 4 here and step 5 and go to [Section: Detection Peak Extraction](## Detection Peak Extraction). But if you want to evaluate the baseline for visual queries 2D localization: Copy `scripts/evaluate_vq.sh` to the experiment directory, update the paths and checkpoint id, and execute it. Note: To evaluate with the particle filter tracker, add the command line argument `tracker.type="pfilter"`.
```bash
EXPT_ROOT=<experiment path>
cp $VQ2D_ROOT/scripts/evaluate_vq.sh $EXPT_ROOT
cd $EXPT_ROOT
<UPDATE PATHS in evaluate_vq.sh>
chmod +x evaluate_vq.sh && ./evaluate_vq.sh
```
We provide pre-trained models for reproducibility. They can be downloaded using the ego4d CLI as follows:
```bash
python -m ego4d.cli.cli -y --output_directory /path/to/output/ --datasets vq2d_models
```

5. Step 4 is for the validation set. Then for the test set of VQ2D: Copy `scripts/get_challenge_predictions.sh` to the experiment directory, update the paths and checkpoint id, and execute it. The arguments are similar to the baseline evaluation in the previous section, but the script has been modified to output predictions consistent with the challenge format.
```bash
EXPT_ROOT=<experiment path>
cp $VQ2D_ROOT/scripts/get_challenge_predictions.sh $EXPT_ROOT
cd $EXPT_ROOT
<UPDATE PATHS in get_challenge_predictions.sh>
chmod +x get_challenge_predictions.sh && ./get_challenge_predictions.sh
```

### Start from our precomputed results
Since EgoLoc doesn't need the tracking result, we only extract the detection result. Please refer to the section below to see how to do it. Also, check `ego_loc/2d_retrieval/data` and [this shared Google Drive folder](https://drive.google.com/drive/folders/1hmCfNT3rolj_llIRc8UUKpaw1Chppv-Y?usp=sharing) for precomputed results.







## Detection Peak Extraction

### Start from our precomputed results
Check `ego_loc/2d_retrieval/data` and [this shared Google Drive folder](https://drive.google.com/drive/folders/1hmCfNT3rolj_llIRc8UUKpaw1Chppv-Y?usp=sharing) for precomputed results.
You'll need the extracted `vq2d_val(test)_vq3d_peak_clip.json`, `baseline_vq3d_detection_v105_val(test).json` and `clips_frames`.



### Start from scratch

#### Validation set
After you have set up the VQ2D environment and checkpoint successfully, run the following code to get detection peaks for the validation set:

```bash
# validation set
# if on a slurm server
bash get_detection_peaks_val.sh
# else run the script locally, you may need to configure your local path
python get_detection_peaks_val.py
```
#### Test set

Similarly, check the following script for the test set. Since Ego4D uses different formats for validation and test set, we process the test set separately here.



```bash
# test set
# if on a slurm server
bash get_detection_peaks_test.sh
# else run the script locally, you may need to configure your local path
python get_detection_peaks_test.py
```



## Acknowledgment

The 2D object retrieval part heavily relies on  Ego4D's VQ2D [official code](https://github.com/EGO4D/episodic-memory) repo.