#!/bin/bash
# (Formerly testfull.sh)

# Configurable base directory
BASE_DIR="$HOME/TestData"
TEMPLATE_DIR="$BASE_DIR/templates"
ATLAS_DIR="$BASE_DIR/atlas"

# Loop through subject IDs
for number in $(seq -w 1 200); do
    subject="sub-$number"
    
    echo "Starting preprocessing for $subject"
    
    anatomical="${subject}_T1w.nii.gz"
    functional="${subject}_run1.nii.gz"

    # Skull stripping the anatomical image
    if [[ ! -f "${subject}_T1w_brain.nii.gz" ]]; then
        bet2 "$anatomical" "${subject}_T1w_brain.nii.gz" -f 0.5
    fi

    # Motion correction for functional image
    mcflirt -in "$functional" -ref SBref

    # Topup correction for susceptibility distortions
    fslmerge -t AP_PA_image A_P P_A
    topup --imain=AP_PA_image --datain=acq_params.txt --config=b02b0.cnf --out=topup_results
    applytopup --imain="$functional" --inindex=1 --topup=topup_results --datain=acq_params.txt --method=jac --out=DC_functional

    # Normalization using ANTs
    antsRegistrationSyNQuick.sh -d 3 -f "${subject}_T1w_brain.nii.gz" -m DC_functional -o func_to_anat
    antsRegistrationSyNQuick.sh -d 3 -f "$TEMPLATE_DIR/template.nii.gz" -m "${subject}_T1w_brain.nii.gz" -o anat_to_template
    antsApplyTransforms -d 3 -i func_to_anat -r "$TEMPLATE_DIR/template.nii.gz" -o func_to_template \
        -t anat_to_template_1Warp.nii.gz -t anat_to_template_GenericAffine.mat -n NearestNeighbor

    # Extract ROI data
    mkdir -p ROI
    for mask in $ATLAS_DIR/*; do
        fslmeants -i func_to_template -o ROI/output_${mask}.txt -m "$mask"
    done

    echo "Processing complete for $subject"
done
