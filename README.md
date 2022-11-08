
# CRIPP-VQA: Counterfactual Reasoning about Implicit Physical Properties via Video Question Answering
This repository contains the codebase for the EMNLP'22 main conference long paper => CRIPP-VQA: Counterfactual Reasoning about Implicit Physical Properties via Video Question Answering

[Demo](https://maitreyapatel.com/CRIPP-VQA/) | [Arxiv](https://arxiv.org/abs/2211.03779/) | [Dataset](https://maitreyapatel.com/CRIPP-VQA/#dataset) 

<!-- <table border="0" style="width:100%">
 <tr>
    <td style="width:60%">
    <h2><b>Abstract</b></h2>
    Videos often capture objects, their visible properties, their motion, and the interactions between different objects. Objects also have physical properties such as mass, which the imaging pipeline is unable to directly capture. However, these properties can be estimated by utilizing cues from relative object motion and the dynamics introduced by collisions. In this paper, we introduce CRIPP-VQA a new video question answering dataset for reasoning about the implicit physical properties of objects in a scene. CRIPP-VQA contains videos of objects in motion, annotated with questions that involve counterfactual reasoning about the effect of actions, questions about planning in order to reach a goal, and descriptive questions about visible properties of objects. The CRIPP-VQA test set enables evaluation under several out-of-distribution settings -- videos with objects with masses, coefficients of friction, and initial velocities that are not observed in the training distribution. Our experiments reveal a surprising and significant performance gap in terms of answering questions about implicit properties (the focus of this paper) and explicit properties of objects (the focus of prior work).
    </td>
    <td style="width:40%">
        <div  align="center">
            <img src="tmp/cripp_main_fig.png" width="500px"/>
        </div>
    </td>
 </tr>
</table> -->

<div  align="center">
    <img src="tmp/cripp_main_fig.png" width="500px"/>
</div>

## Dataset preparation
* Download the annotations and question-answer pair files from [dataset link](https://maitreyapatel.com/CRIPP-VQA/#dataset).
* Follow the instructions from [datasets.md](dataset/datasets.md) to generate the video instances from the annotations. 
* Alternatively, download the per-processed Mask-RCNN based features from  [feature link](https://maitreyapatel.com/CRIPP-VQA/#dataset).

## Aloe*+BERT Model
The Aloe*+BERT is PyTorch version of the modified baseline Aloe from [Ding et. al.](https://openreview.net/forum?id=lHmhW2zmVN)

* Please refer to the [modeling.md](Aloe-star/README.md) for the instructions on training of the Aloe*+BERT.

## Evaluation
* Evaluations for the descriptive and counterfactual questions are straightforward.
* For planning based task evaluation, please refer to the [evaluations.md](evaluation/evaluations.md) for step by step instructions.

## Acknowledgement
This work is supported by NSF and DARPA projects. We also thank the David Ding for timely feedback to reproduce the results of PyTorch version of the Aloe on CLEVRER dataset. 

## Citation
Please consider citing the paper if you find it relevant or useful. 
```
@inproceedings{patel2022cripp,
    title = "{CRIPP-VQA}: Counterfactual Reasoning about Implicit Physical Properties via Video Question Answering",
    author = " Patel, Maitreya and 
        Gokhale, Tejas and 
        Baral, Chitta and
        Yang, Yezhou",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    year = "2022",
}
```

## Issues
For technical concerns please create the GitHub issues. A quick way to resolve any issues would be to reach out to the author at [maitreya.patel@asu.edu](mailto:maitreya.patel@asu.edu).