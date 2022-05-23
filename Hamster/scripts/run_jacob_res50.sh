model=resnet50
array=( 1,2 3,4 5,6 7,8 9,10 )
device_idx=2

for sample_idx in ${array[@]}
do
echo ${device_idx}
CUDA_VISIBLE_DEVICES=${device_idx} nohup python jacobian.py -m ${model} --wo-pretrained --sample-idx ${sample_idx} > logs/jacob_${model}_${sample_idx}_wo_pretrained.log &
device_idx=$[device_idx + 1]
done

CUDA_VISIBLE_DEVICES=7 python extract_perturb.py -m ${model} --wo-pretrained > logs/feat_${model}_wo_pretrained.log
