# Our Loss Version 1 for Duke
python3 tools/train.py --config_file='configs/softmax_triplet_Ours_V1.yml'\
 OUTPUT_DIR "/data/han.sun/Checkpoints/ReID_Strong_BL/Duke" \
  LOG_NAME "log_test.txt"\
  OURS.ALPHA "18.0" OURS.BETA "0.5" MODEL.DEVICE_ID "'1'" MODEL.ADJUST_LR "off"\
  MODEL.METRIC_LOSS_TYPE "ours" DATALOADER.SAMPLER "ours" MODEL.NECK "APE"\
  DATASETS.NAMES "'dukemtmc'"  INPUT.RE_PROB "0.5"  MODEL.LAST_STRIDE "1"\
  DATALOADER.NUM_INSTANCE "8" SOLVER.BASE_LR "3.5e-4" SOLVER.WARMUP_ITERS "0" MODEL.IF_TRIPLET "no"