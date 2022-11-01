# Dataset Preparation
This folder contains quick guidelines about the proposed CRIPP-VQA dataset.

## Instructions
As the video files are too large to manage on cloud platforms, we provide only annotations files and video features. However, to generate the videos from scratch, please follow the below guidelines. This will benefit those who need raw visual data.

Note: Although we provide annotations for all counterfactual actions, the CRIPP-VQA challenge does not utilize videos or annotations of counterfactual scenarios during training/testing.  But feel free to use this script to play around with different experiments. 

### Download
* Annotation files are available at [link](https://maitreyapatel.com/CRIPP-VQA/#dataset).
* [Optional] Mask-RCNN features are also available at [link](https://maitreyapatel.com/CRIPP-VQA/#dataset). 

### Video Generation
Follow the below steps to generate the video from the ***i.i.d.*** annotation files:

```bash
# setup repo
git clone git@github.com:Maitreyapatel/CRIPP-VQA.git

# setup env
cd CRIPP-VQA/dataset
virtualenv venv
source ./venv/bin/activate
pip install -r requirements.txt
```
If there is an issue with *TDW* installation, please refer to the official docs at [GitHub](https://github.com/threedworld-mit/tdw).

We assume that annotation files are extracted inside the `CRIPP-VQA/dataset/annotations/`. And the generated videos will be stored inside `CRIPP-VQA/dataset/generated_videos`.

```bash
source ./venv/bin/activate
sh ./image_generation.sh <folder-name-inside-annotation-folder>
```

Note: Feel free to modify the `image_generation.sh` script according to your use case. Having errors !? (look below)


## Issues
For technical concerns please create the GitHub issues. A quick way to resolve any issues would be to reach out to the author at [maitreya.patel@asu.edu](mailto:maitreya.patel@asu.edu).