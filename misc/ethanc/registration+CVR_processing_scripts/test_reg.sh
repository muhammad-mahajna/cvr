#!/bin/bash


#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=FLIRT_TEST_2
#SBATCH --mail-type=END
#SBATCH --mail-user=ethan.church2@mail.mcgill.ca
#SBATCH --output=OutputFrom_Test_Reg_%j.out

module load fsl/6.0.0
module load ants/2.5.0
cd ~/pre_ml/func

func="SF_01035_rsBOLD_fc.nii.gz"
T1="../struc/SF_01035_T1.nii"
fslroi $func mean 0 1


#flirt -in processing/SF_01035_rsBOLD_fc_frame.nii.gz -ref ~/pre_ml/struc/SF_01035_brain.nii.gz -omat processing/test_ref_2_t1 -out test_ref_2_t1

#flirt -in SF_01035_rsBOLD_fc -applyxfm -init test_ref_2_t1 -ref SF_01035_rsBOLD_fc_frame.nii.gz -out registered/test_outpu

antsRegistrationSyNQuick.sh -d 3 \
	-f $T1 \
	-m mean.nii.gz \
	-o output_prefix_ \
	-t r 

antsApplyTransforms -d 3 \
	-e 3 \
	-i $func \
	-r mean.nii.gz \
    	-o boldToFixedDeformed.nii.gz \
    	-t output_prefix_0GenericAffine.mat \
	-v 1


