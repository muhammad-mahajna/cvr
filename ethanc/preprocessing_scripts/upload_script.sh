#!/bin/bash
dir=$1

cd ~/TestData/RAW_DATA/undone/$dir/images
scp ${dir}_T1.nii ethan.church@arc.ucalgary.ca:~/pre_ml/struc
cd processing
scp ${dir}_brain.nii.gz ethan.church@arc.ucalgary.ca:~/pre_ml/struc
scp rsBOLD_fc.nii.gz ethan.church@arc.ucalgary.ca:~/pre_ml/func/unprocessed/${dir}_rsBOLD_fc.nii.gz
echo "tranfer completed for $dir"


