#!/bin/bash


#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=FLIRT_TEST_2
#SBATCH --mail-type=END
#SBATCH --mail-user=ethan.church@ucalgary.ca
#SBATCH --output=OutputFrom_Registration_%j.out

module load fsl/6.0.0
module load ants

subjects=$(ls ~/pre_ml/CVR_MAPS | sed -r 's/(SF_[0-9]+).*/\1/' | sort | uniq)
for sub in $subjects
do
        cd ~/pre_ml/CVR_MAPS
        T1="../struc/${sub}_brain.nii.gz"
        CVR="Thresholded/${sub}_BOLD_CVR_thresh.nii.gz"
        antsRegistrationSyNQuick.sh -d 3 -f $T1 -m $CVR -o processing/${sub}_transform_ -t r
      antsApplyTransforms -d 3 -i $CVR -r ../func/processing/${sub}_frame.nii.gz -o registered/${sub}_CVR_2_T1.nii.gz -t processing/${sub}_transform_0GenericAffine.mat	
        
        echo "registration completed for $sub"

done
