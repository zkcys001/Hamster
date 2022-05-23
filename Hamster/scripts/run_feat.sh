# CUDA_VISIBLE_DEVICES=0 nohup python extract_perturb.py -m resnet18 --wo-pretrained > logs/feat_resnet18_wo_pretrained.log &

CUDA_VISIBLE_DEVICES=6 python extract_perturb.py -m resmlp --wo-pretrained
CUDA_VISIBLE_DEVICES=7 python extract_perturb.py -m gmlpmixer_t --wo-pretrained
