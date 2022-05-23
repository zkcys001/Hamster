model=resnet18
array=( 1,50 51,100 )
device_idx=0

for sample_idx in ${array[@]}
do
echo ${device_idx}
CUDA_VISIBLE_DEVICES=${device_idx} nohup python jacobian.py -m ${model} --wo-pretrained --sample-idx ${sample_idx} > logs/jacob_${model}_${sample_idx}_wo_pretrained.log &
device_idx=$[device_idx + 1]
done

CUDA_VISIBLE_DEVICES=0 python extract_perturb.py -m ${model} --wo-pretrained > logs/feat_${model}_wo_pretrained.log
