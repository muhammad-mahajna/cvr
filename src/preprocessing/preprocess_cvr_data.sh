#!/bin/bash
# (Formerly pre_cvr.sh)

# Configurable base directory
RAW_DATA_DIR="../RAW_DATA"

# Preprocess CVR data
cd "$RAW_DATA_DIR"
for dir in $(ls | grep -i "SF_01043"); do 
    cd "$dir"
    
    # Convert DICOM to NIfTI
    dcm2niix -f "${dir}_%f" -o ./images/ ./MBME_CO2_O2

    # Perform motion correction
    mcflirt -in *MBME_CO2_O2_e1.nii -out ./processing/echo1_mc
    mcflirt -in *MBME_CO2_O2_e2.nii -out ./processing/echo2_mc
done
