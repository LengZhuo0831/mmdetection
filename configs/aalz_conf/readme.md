

### train

res50+dyhead+atss
```shell
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29501 ./tools/dist_train.sh configs/aalz_conf/atss_r50_fpn_dyhead_1x_coco.py 4 --auto-scale-lr --auto-resume

CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=29502 ./tools/dist_train.sh configs/aalz_conf/atss_r50_fpn_dyhead_1x_coco_score90_nms50.py 4 --auto-scale-lr --auto-resume
```

