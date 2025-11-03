# #!/bin/bash

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_ann_msc_$i"
#     yaml_file="yaml/Jeon/folds/ann/Jeon_ann_msc_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/ann/Jeon_ann_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_ann_msc_$i"
#     yaml_file="yaml/Jeon/folds/ann/Jeon_ann_msc_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/ann/Jeon_ann_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_Improved_Incep_msc_$i"
#     yaml_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_msc_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_Improved_Incep_msc_$i"
#     yaml_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_msc_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_trans_msc_$i"
#     yaml_file="yaml/Jeon/folds/trans/Jeon_trans_msc_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/trans/Jeon_trans_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_trans_msc_$i"
#     yaml_file="yaml/Jeon/folds/trans/Jeon_trans_msc_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/trans/Jeon_trans_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_ann_snv_$i"
#     yaml_file="yaml/Jeon/folds/ann/Jeon_ann_snv_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/ann/Jeon_ann_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_ann_snv_$i"
#     yaml_file="yaml/Jeon/folds/ann/Jeon_ann_snv_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/ann/Jeon_ann_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_Improved_Incep_snv_$i"
#     yaml_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_snv_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_Improved_Incep_snv_$i"
#     yaml_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_snv_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_trans_snv_$i"
#     yaml_file="yaml/Jeon/folds/trans/Jeon_trans_snv_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/trans/Jeon_trans_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_trans_snv_$i"
#     yaml_file="yaml/Jeon/folds/trans/Jeon_trans_snv_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/trans/Jeon_trans_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_ann_sg_$i"
#     yaml_file="yaml/Jeon/folds/ann/Jeon_ann_sg_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/ann/Jeon_ann_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_ann_sg_$i"
#     yaml_file="yaml/Jeon/folds/ann/Jeon_ann_sg_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/ann/Jeon_ann_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_Improved_Incep_sg_$i"
#     yaml_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_sg_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_Improved_Incep_sg_$i"
#     yaml_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_sg_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/Improved_Incep/Jeon_Improved_Incep_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Jeon_trans_sg_$i"
#     yaml_file="yaml/Jeon/folds/trans/Jeon_trans_sg_${i}.yaml"
#     hyp_file="yaml/Jeon/folds/trans/Jeon_trans_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
Loop over fold numbers 1 to 5
for i in {2..5}
do
    exp_name="Jeon_trans_sg_$i"
    yaml_file="yaml/Jeon/folds/trans/Jeon_trans_sg_${i}.yaml"
    hyp_file="yaml/Jeon/folds/trans/Jeon_trans_hyp.yaml"
    checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
    echo "Running testing for $exp_name"
    python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_ann_msc_$i"
#     yaml_file="yaml/Leone/folds/ann/Leone_ann_msc_${i}.yaml"
#     hyp_file="yaml/Leone/folds/ann/Leone_ann_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_ann_msc_$i"
#     yaml_file="yaml/Leone/folds/ann/Leone_ann_msc_${i}.yaml"
#     hyp_file="yaml/Leone/folds/ann/Leone_ann_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_Improved_Incep_msc_$i"
#     yaml_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_msc_${i}.yaml"
#     hyp_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_Improved_Incep_msc_$i"
#     yaml_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_msc_${i}.yaml"
#     hyp_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_trans_msc_$i"
#     yaml_file="yaml/Leone/folds/trans/Leone_trans_msc_${i}.yaml"
#     hyp_file="yaml/Leone/folds/trans/Leone_trans_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_trans_msc_$i"
#     yaml_file="yaml/Leone/folds/trans/Leone_trans_msc_${i}.yaml"
#     hyp_file="yaml/Leone/folds/trans/Leone_trans_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_ann_snv_$i"
#     yaml_file="yaml/Leone/folds/ann/Leone_ann_snv_${i}.yaml"
#     hyp_file="yaml/Leone/folds/ann/Leone_ann_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_ann_snv_$i"
#     yaml_file="yaml/Leone/folds/ann/Leone_ann_snv_${i}.yaml"
#     hyp_file="yaml/Leone/folds/ann/Leone_ann_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_Improved_Incep_snv_$i"
#     yaml_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_snv_${i}.yaml"
#     hyp_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_Improved_Incep_snv_$i"
#     yaml_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_snv_${i}.yaml"
#     hyp_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_trans_snv_$i"
#     yaml_file="yaml/Leone/folds/trans/Leone_trans_snv_${i}.yaml"
#     hyp_file="yaml/Leone/folds/trans/Leone_trans_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_trans_snv_$i"
#     yaml_file="yaml/Leone/folds/trans/Leone_trans_snv_${i}.yaml"
#     hyp_file="yaml/Leone/folds/trans/Leone_trans_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_ann_sg_$i"
#     yaml_file="yaml/Leone/folds/ann/Leone_ann_sg_${i}.yaml"
#     hyp_file="yaml/Leone/folds/ann/Leone_ann_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_ann_sg_$i"
#     yaml_file="yaml/Leone/folds/ann/Leone_ann_sg_${i}.yaml"
#     hyp_file="yaml/Leone/folds/ann/Leone_ann_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_Improved_Incep_sg_$i"
#     yaml_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_sg_${i}.yaml"
#     hyp_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_Improved_Incep_sg_$i"
#     yaml_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_sg_${i}.yaml"
#     hyp_file="yaml/Leone/folds/Improved_Incep/Leone_Improved_Incep_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================

# # Training ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_trans_sg_$i"
#     yaml_file="yaml/Leone/folds/trans/Leone_trans_sg_${i}.yaml"
#     hyp_file="yaml/Leone/folds/trans/Leone_trans_hyp.yaml"
    
#     echo "Running training for $exp_name"
#     python train.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file"
# done
# # Training End ==================================================

# # Test start ==================================================
# # Loop over fold numbers 1 to 5
# for i in {2..5}
# do
#     exp_name="Leone_trans_sg_$i"
#     yaml_file="yaml/Leone/folds/trans/Leone_trans_sg_${i}.yaml"
#     hyp_file="yaml/Leone/folds/trans/Leone_trans_hyp.yaml"
#     checkpoint_path="logFile/$exp_name/weights/best_checkpoint.pth"
    
#     echo "Running testing for $exp_name"
#     python test.py --exp_name "$exp_name" --yaml "$yaml_file" --hyp "$hyp_file" --checkpoint "$checkpoint_path"
# done
# # Test end ==================================================



