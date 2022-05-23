model=vit_t
array=( 1,2 3,4 5,6 7,8 9,10 )
device_idx=0

for sample_idx in ${array[@]}
do
echo ${device_idx}
CUDA_VISIBLE_DEVICES=${device_idx} nohup python jacobian.py -m ${model} --wo-pretrained --sample-idx ${sample_idx} > logs/jacob_${model}_${sample_idx}_wo_pretrained.log &
device_idx=$[device_idx + 1]
done

# CUDA_VISIBLE_DEVICES=5 python extract_perturb.py -m ${model} --wo-pretrained > logs/feat_${model}_wo_pretrained.log

CUDA_VISIBLE_DEVICES=5 python extract_perturb.py -m vit_t
CUDA_VISIBLE_DEVICES=6 python extract_perturb.py -m vit_t --wo-pretrained
