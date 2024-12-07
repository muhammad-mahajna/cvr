#!/bin/bash

#Iterate through the files
for number in $(sed -w 1 200)
do
	subject= "sub-$number"
	
	anatomical= "${subject}_T1w.nii.gz"
	functional= "${subject}_run1.nii.gz"

	#skullstripping both the anatomical image if not already done 
	if [[ -f ${subject}_T1w_brain.nii.gz]]
	then
		bet2 anatomical ${subject}_T1w_brain.nii.gz -f 0.5
	fi
	
	#running mcflirt for motion correction on the functional image, outfile is the name appended with _mcf
	#if there is no reference file
	mcflirt -in functional -ref SBref

	#using topup to correct for any artifacts or susceptibility 
	#first merging the two files into one
	fslmerge AP_PA_image A_P P_A 

	topup --imain=AP_PA_images --datain=acq_params.txt --conffig=b02b0.cnf --out=topup_results

	#myresults will be an estimate of the field distortions in every voxel
	#Now applying the estimate to correct the distortion

	applytopup --imain=functional --inindex=1 --topup=topup_results --datain=acq_params.txt --method=jac --out=DC_functional
	#now should have a functional image that has had the susceptibility induced resonance intereference reduced
	
	#now should Normalize the brain to whatever the 90 ROI thing has been set to in order to use the atlas.
	#using the ants tools: 1. normalize func to anat 2. normalize anat to template 3. use same transformations as (2) to
	#transform func to template

	#1
	antsRegistrationSyNQuick.sh -d 3 -f skullstripped_anat -m DC_functional -o func_to_anat
	
	#should output 5 files 

	#2
	antsRegistrationSyNQuick.sh -d 3 -f template -m skulstripped_anat -o anat_to_template

	#3, remember -t's are read right to left so want affline first then warp
	antsApplyTransforms -d 3 -i func_to_anat -r template -o func_to_template -t anat_to_template_1Warp.nii.gz \
		-t anat_to_template_GenericAffine.mat  	-n NearestNeighbor

	#now should have both anatomical and functional normalized to whatever template you are using 
	#now have to do ROI for loops, right? like for each ROI on the atlas create and label a mean time series, pretty sure I can
	#use the fslmeants to do that right? because each one would be like a mask that I could input into fslmeants
	#may need a directory of 3d maps to do this we can see i guess

	# -o can help me with outputting a text matrix, could also redirect output into a different file?
	#could maybe use absolute path in getting the
       	mkdir ROI 	
	for mask in atlas
	do
		fslmeants -i func_to_template_1Warp.nii.gz -o ROI/output.txt -m $mask
	done
done	
	

