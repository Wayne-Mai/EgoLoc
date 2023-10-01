# Video 2D Object Retrieval

This module is mostly adapted from Ego4D's VQ2D [official code](https://github.com/EGO4D/episodic-memory).

We remove the tracker of the pretrained model, then use the detector to select those peak predictions.

We provide our pre-computer 2D results in `data/vq2d_val_vq3d_peak_only.json`.
You can use your own 2D prediction result to replace ours, and don't forget to follow the same format.

In case you want to reproduce the 2D results by yourself, please kindly follow instructions below:


## TODO

- [x] Init Code
- [ ] Merge 2D retrieval code for validation and test
- [ ] Add code for test set also


## Environment

Setup the environment as Ego4D VQ2D(https://github.com/EGO4D/episodic-memory) requested:



1. Clone the repository from [here](https://github.com/EGO4D/episodic-memory).
    ```
    git clone git@github.com:EGO4D/episodic-memory.git
    cd episodic-memory/VQ2D
    export VQ2D_ROOT=$PWD
    ```
2. Create conda environment.
    ```
    conda create -n ego4d_vq2d python=3.8
    ```

3. Install [pytorch](https://pytorch.org/) using conda. We rely on cuda-10.2 and cudnn-7.6.5.32 for our experiments.
    ```
    module load cuda/10.2.89
    module load cudnn/7.6.5.32
    conda install pytorch==1.8.1 torchvision==0.9.1 torchaudio==0.8.1 cudatoolkit=10.2 -c pytorch

    conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch

    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
    ```

4. Install additional requirements using `pip`.
    ```
    pip install -r requirements.txt

    pip install -r requirements_new.txt
    ```

5. Install [detectron2](https://github.com/facebookresearch/detectron2).
   
    ```
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
    
    python -m pip install detectron2 -f   https://dl.fbaipublicfiles.com/detectron2/wheels/cu102/torch1.8/index.html

    python -m pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html

    python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
    ```

6.  Install pytracking according to [these instructions](https://github.com/visionml/pytracking/blob/master/INSTALL.md). Download the pre-trained [KYS tracker weights](https://drive.google.com/drive/folders/1WGNcats9lpQpGjAmq0s0UwO6n22fxvKi) to `$VQ2D_ROOT/pretrained_models/kys.pth`.
    ```
    cd $VQ2D_ROOT/dependencies
    git clone git@github.com:visionml/pytracking.git
    git checkout de9cb9bb4f8cad98604fe4b51383a1e66f1c45c0
    ```

    Note: For installing the [spatial-correlation-sampler](https://github.com/ClementPinard/Pytorch-Correlation-extension) dependency for pytracking, follow these steps if the pip install fails.
    ```
    cd $VQ2D_ROOT/dependencies
    git clone git@github.com:ClementPinard/Pytorch-Correlation-extension.git
    cd Pytorch-Correlation-extension
    python setup.py install
    ```


## Data

For VQ3D task, you don't need to download VQ2D full videos but just the pretrained checkpoint. The pretrained model can be downloaded using the ego4d CLI as follows:

 ```
    python -m ego4d.cli.cli -y --output_directory /path/to/output/ --datasets vq2d_models
```
Up to now, you should have followed the main page's readme to download VQ3D video clips in `$VQ3D_ROOT` folder and prepared Ego4D's VQ2D code in `$VQ2D_ROOT` folder.

We recommend to organize the file structure as follows:

```
- ego_loc
    - 2d_retrieval
        - [vq2d](Ego4D's VQ2D baseline code)
            - pretrained_models
        - [data](soft link to vq3d/data)
    - vq3d
        - data
            - clips
```





## Detection Peak Extraction


After you have setup the VQ2D environment and checkpoint successfully, run this script to get similarity score for VQ3D video clips:
```
bash get_similarity_score_val.sh
```

Then run following code to get detection peaks.

```
# validation set
bash get_similarity_score_val.py
python get_detection_peaks_val.py
```