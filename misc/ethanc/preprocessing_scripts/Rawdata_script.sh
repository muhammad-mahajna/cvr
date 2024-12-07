#!/bin/bash

dir=$1
cd ~/TestData/RAW_DATA/undone 
cd $dir 
pwd
mkdir images
dcm2niix -f "${dir}_rsBOLD_ET" -o ./images/ ./*-rsBOLD-End-tidal
dcm2niix -f "${dir}_rsBOLD_ET_FLIP" -o ./images/ ./*-rsBOLD-End-tidal-_Flip_PE
dcm2niix -f "${dir}_T1" -o ./images ./*-SAG_FSPGR_BRAVO
#dcm2niix -f "${dir}_%f" -o ./images/ ./MBME_CO2_O2
cd images
mkdir processing
bet2 *T1.nii ./processing/${dir}_brain -f 0.65
mcflirt -in *rsBOLD_ET.nii -out ./processing/rsBOLD_mc
mcflirt -in *rsBOLD_ET_FLIP.nii -out ./processing/rsBOLD_FLIP_mc
#mcflirt -in *MBME_CO2_O2_e1.nii -out ./processing/echo1_mc
#mcflirt -in *MBME_CO2_O2_e2.nii -bet2 *T1.nii ./processing/${dir}_brainout ./processing/echo2_mc


touch acqparams.txt
for file in $(ls *rsBOLD*.json)
do
	json_file="$file"
	phase_encoding=$(jq -r '.PhaseEncodingDirection' "$json_file")
	readout_time=$(jq -r '.TotalReadoutTime' "$json_file")
	if [[ "$phase_encoding" == "j-" ]]; then
		phase_vector="0 -1 0"
	else 
		phase_vector="0 1 0"
	fi
	echo "$phase_vector $readout_time" >> acqparams.txt
done 
mv acqparams.txt ./processing
cd processing


#mkdir segmentation
#fast -t 1 -n 3 -H 0.1 -I 4 -l 20.0 -o ./segmentation/T1_brain ${dir}_brain.nii.gz
#fslmaths ./segmentation/T1_brain_pve_1.nii.gz -thr 0.95 -bin ./T1_brain_gm_mask.nii.gz


fslroi rsBOLD_mc rsBOLD_ff 0 1
fslroi rsBOLD_FLIP_mc rsBOLD_FLIP_ff 0 1
fslmerge -t merged_epi rsBOLD_ff rsBOLD_FLIP_ff
	
topup --imain=merged_epi --datain=acqparams.txt --config=b02b0.cnf --out=topup_results
applytopup --imain=rsBOLD_mc --inindex=1 --topup=topup_results --datain=acqparams.txt --method=jac --out=rsBOLD_fc
	
	#fslroi rsBOLD_fc rsBOLD_fc_frame 0 1
	#flirt -in rsBOLD_fc_frame -ref ${dir}_brain -out ref_2_t1 -omat ref_2_t1.mat
	#flirt -in rsBOLD_fc -ref ${dir}_brain -applyxfm -init ref_2_t1.mat -out rsBOLD_2_t1
	#fslmeants -i rsBOLD_2_t1 -o gm_meants.txt -m ./segmentation/T1_brain_gm_mask.nii.gz
cd ../../../..
pwd
echo "done preprocessing"


