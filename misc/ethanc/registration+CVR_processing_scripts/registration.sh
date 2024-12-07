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

subjects=$(ls ~/pre_ml/func/unprocessed | sed -r 's/(SF_[0-9]+).*/\1/' | sort | uniq)
for sub in $subjects
do 
	cd ~/pre_ml/func/
	T1="../struc/${sub}_brain.nii.gz"
	func="unprocessed/${sub}_rsBOLD_fc.nii.gz"
	fslroi $func processing/${sub}_frame 0 1 
	antsRegistrationSyNQuick.sh -d 3 -f $T1 -m processing/${sub}_frame.nii.gz -o processing/${sub}_ref2T1_ -t r

	antsApplyTransforms -d 3 -e 3 -i $func -r processing/${sub}_frame.nii.gz -o registered/${sub}_2_T1.nii.gz -t processing/${sub}_ref2T1_0GenericAffine.mat
	mv $func completed
	echo "registration completed for $sub"

done
