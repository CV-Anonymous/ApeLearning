APE Learning for Person Re-identification
=========

The codes are expanded on a [Re-ID Strong Baseline](https://github.com/michuanhaohao/reid-strong-baseline), which is open sourced by Hao Luo.

Installation
---------
[pytorch](https://pytorch.org)    
[ignite=0.1.2](https://github.com/pytorch/ignite)   
[yacs](https://github.com/rbgirshick/yacs)

Data
---------
We use GMS and RPE learning on three Re-ID datasets:    
[Market1501](http://www.liangzheng.com.cn/Project/project_reid.html)    
[DukeMTMC](https://github.com/layumi/DukeMTMC-reID_evaluation#download-dataset)   
[CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html)    

Evaluation
---------
The evaluation criteria Market1501 and DukeMTMC are provided by the baseline.     
We use the new training/test protocal for CUHK03 ([CUHK03-NP](https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP)), which can calculate mAP and CMC similar to the above two datasets.

Train
---------
You can run the `./*.sh` files according to the different datasets directly or run these codes in the `.sh` files as follows:
1. Market1501   
(1) GMS Loss    

(2) APE Loss
``` python
python3 tools/train.py --config_file='configs/softmax_triplet_Ours_V1.yml'  OUTPUT_DIR "/data/han.sun/Checkpoints/ReID_Strong_BL/Market1501" LOG_NAME "log_test.txt" OURS.ALPHA "20.0" OURS.BETA "0.5" MODEL.DEVICE_ID "'0'" MODEL.ADJUST_LR "off"   MODEL.METRIC_LOSS_TYPE "ours" DATALOADER.SAMPLER "ours" MODEL.NECK "APE" DATASETS.NAMES "'market1501'"  INPUT.RE_PROB "0.7"  MODEL.LAST_STRIDE "1" DATALOADER.NUM_INSTANCE "8" SOLVER.BASE_LR "3.5e-4" SOLVER.WARMUP_ITERS "0" MODEL.IF_TRIPLET "no"
```

2. DukeMTMC   
(1) GMS Loss

(2) APE Loss

3. CUHK03-Detected    
(1) GMS Loss

(2) APE Loss

4. Main Parameters in training

Illustration
---------
The baseline has some authors information in their codes, who have no relationships with us. For reviewing the paper, we comment these information, and we will recover them after reviewing. If we miss some information (e.g., name and email) in the codes, we declare that the information is not belong to any author in our paper.
