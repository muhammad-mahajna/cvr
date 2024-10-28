#!/bin/bash

#first convert the raw files into dcm
cd ..
cd RAW_DATA
mkdir "$1_nifti"
newdir="$1_nifti"

dcm2niix -o /home/ethanchurch/TestData/$newdir/ /home/ethanchurch/TestData/$1/3-*
dcm2niix -o /home/ethanchurch/TestData/$newdir/ /home/ethanchurch/TestData/$1/22-*
dcm2niix -o /home/ethanchurch/TestData/$newdir/ /home/ethanchurch/TestData/$1/23-*


cd $newdir
mkdir processing
bet2 3*.nii ./processing/anat_image

mcflirt -in 22*.nii -out ./processing/22_mc
mcflirt -in 23*.nii -out ./processing/23_mc

touch acqparams.txt
for num in $(seq 22 23)
do 

	json_file=$(ls $num*.json)
	phase_encoding=$(jq -r '.PhaseEncodingDirection' "$json_file")
	readout_time=$(jq -r '.TotalReadoutTime' "$json_file")
	if [[ "$phase_encoding" == "j-" ]]; then
		phase_vector="0 -1 0"
	       echo "$num spin is neg"	
	else 
		phase_vector="0 1 0"
		echo "$num spin is pos" 
	fi 
	echo "$phase_vector $readout_time" >> acqparams.txt
done
mv acqparams.txt processing
cd processing

fslroi 22_mc 22_first_frame 0 1
fslroi 23_mc 23_first_frame 0 1
fslmerge -t 22_23_merged 22_first_frame 23_first_frame

#now topup
topup --imain=22_23_merged --datain=acqparams.txt --config=b02b0.cnf --out=topup_results
#applying
applytopup --imain=22_mc --inindex=1 --topup=topup_results --datain=acqparams.txt --method=jac --out=22_fc





