#!/bin/bash
for i in {2..5}
do
    exp_name="Jeon_ann_pre_$i"
    yaml_file="yaml/Jeon/folds/ann/Jeon_ann_pre_${i}.yaml"
    hyp_file="yaml/Jeon/folds/ann/Jeon_ann_pre_hyp.yaml"
    
    echo "Running training for $exp_name"
    python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
done
for i in {2..5}
do
    exp_name="Jeon_trans_pre_$i"
    yaml_file="yaml/Jeon/folds/trans/Jeon_trans_pre_${i}.yaml"
    hyp_file="yaml/Jeon/folds/trans/Jeon_trans_pre_hyp.yaml"
    
    echo "Running training for $exp_name"
    python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
done