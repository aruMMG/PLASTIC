#!/bin/bash

# Training ==================================================
# # Loop over fold numbers 1 to 5
for i in {1..5}
do
    exp_name="Edi_MIR_Improved_Incep_NoSE_$i"
    yaml_file="yaml/Edi_MIR/Improved_Incep/raw/Improved_Incep_${i}_pre.yaml"
    hyp_file="yaml/Edi_MIR/Improved_Incep/raw/Improved_Incep_hyp.yaml"
    
    echo "Running training for $exp_name"
    python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
done
# Training End ==================================================

# Test start ==================================================
# Loop over fold numbers 1 to 5
# for i in {1..5}
# do
#     exp_name="Edi_MIR_ann_ln_$i"
#     yaml_file="yaml/Edi_MIR/ann_pre/raw/ann_${i}.yaml"
#     hyp_file="yaml/Edi_MIR/ann_pre/raw/ann_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# Test end ==================================================

