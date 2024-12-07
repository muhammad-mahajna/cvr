#!/bin/bash

cd ../RAW_DATA/undone
for dir in $(ls)
do 
	~/TestData/script_tests/Rawdata_script.sh $dir 
	echo "processing completed for $dir"
	~/TestData/script_tests/upload_script.sh $dir
	mv $dir ~/TestData/RAW_DATA/completed
done

	
