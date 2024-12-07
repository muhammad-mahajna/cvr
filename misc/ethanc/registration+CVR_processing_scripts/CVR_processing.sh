#!/bin/bash

module load fsl/6.0.0
cd ~/pre_ml/CVR_MAPS
#first iterate through each file in the CVR_directory
for file in $(ls ~/pre_ml/CVR_MAPS)
do
	if [[ -f $file ]]
	then
		fslroi $file ${file}_resized 0 -1 0 -1 6 26
		fslmaths ${file}_resized -uthr 0.7 ~/pre_ml/CVR_MAPS/Thresholded/${file}_thresh
	fi
done
echo "completed processing for $file"



