#!/bin/bash
cd ../RAW_DATA
for dir in $(ls | grep -i "SF_01043")
do 
	cd $dir 
	dcm2niix -f "${dir}_%f" -o ./images/ ./MBME_CO2_O2
	cd images
	mcflirt -in *MBME_CO2_O2_e1.nii -out ./processing/echo1_mc
	mcflirt -in *MBME_CO2_O2_e2.nii -out ./processing/echo2_mc
done
