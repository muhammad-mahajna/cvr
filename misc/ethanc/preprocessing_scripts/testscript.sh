#!/bin/bash

#have to know what the input files look like to use sed on them

for id in $(seq -w 1 26) 
do 
	subject = "sub-$id"
	echo "starting processing of $id"
	cd $subject

	#if we cannot find a skull-stripped brain we will have to create one from the anatomical image:
	if [[ -f anat/${subject}_T1w_brain_f02.nii.gz ]]
	then 
		bet2 anat/${subject}_T1w.nii.gz anat/${subject}_T1w_brain_f02.nii.gz -f 0.2
	fi

	#copying the design files for the feat analysis
	#must change the initial subject number used in the files to the current subject number
	cp ../design_run1.fsf

	sed -i '' "s|sub-01|${subject}|g" design_run1.fsf


	#now running feat on the data
	echo "running feat on $subject"
	feat design_run1.fsf

	cd..
done

	
