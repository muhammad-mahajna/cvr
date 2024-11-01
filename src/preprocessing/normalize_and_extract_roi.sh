#!/bin/bash

# Example usage:
# ./normalize_and_extract_roi.sh /data/project1 SUBJECT001
# ./normalize_and_extract_roi.sh /data/project1 all

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <base_directory> <subject_id> or 'all' to process all subjects"
    exit 1
fi

# Configurable base directories
BASE_DIR="$1"
SUBJECT_ID="$2"

MNI_TEMPLATE="${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz"
ATLAS_ROI_PATH="${FSLDIR}/data/atlases/HarvardOxford/HarvardOxford-cort-maxprob-thr50-1mm.nii.gz"

# Function to normalize and extract ROI time series for a single subject
normalize_and_extract_roi() {
    SUBJECT_ID="$1"
    PREPROCESSING_DIR="$BASE_DIR/$SUBJECT_ID/preprocessing_results"
    ANTS_RESULTS_DIR="$BASE_DIR/$SUBJECT_ID/ants_results"
    OUTPUT_DIR="$BASE_DIR/$SUBJECT_ID/roi_results"
    LOG_FILE="$OUTPUT_DIR/${SUBJECT_ID}_roi_log.txt"

    # Ensure that the output directory exists
    mkdir -p "$OUTPUT_DIR"

    # Define paths to the required input files
    T1_IMAGE="$ANTS_RESULTS_DIR/${SUBJECT_ID}_ants_Warped.nii.gz"  # Normalized T1 image from ants_results
    FMRI_IMAGE="$PREPROCESSING_DIR/rsBOLD_field_corrected.nii.gz"  # Field-corrected fMRI from preprocessing_results

    # Check if the required input files are available
    if [ ! -f "$T1_IMAGE" ]; then
        echo "Warped T1 image not found for subject $SUBJECT_ID: $T1_IMAGE" | tee -a "$LOG_FILE"
        return 1
    fi
    if [ ! -f "$FMRI_IMAGE" ]; then
        echo "Field-corrected fMRI image not found for subject $SUBJECT_ID: $FMRI_IMAGE" | tee -a "$LOG_FILE"
        return 1
    fi

    # Initialize log file
    echo "Starting ROI extraction for subject $SUBJECT_ID at $(date)" > "$LOG_FILE"

    # Apply T1-to-MNI transformation to fMRI data
    echo "Applying normalization to MNI space for subject $SUBJECT_ID..." | tee -a "$LOG_FILE"
    antsApplyTransforms -d 3 -e 3 -i "$FMRI_IMAGE" -r "$MNI_TEMPLATE" \
        -o "$OUTPUT_DIR/${SUBJECT_ID}_fMRI_normalized.nii.gz" \
        -t "$ANTS_RESULTS_DIR/${SUBJECT_ID}_ants_1Warp.nii.gz" \
        -t "$ANTS_RESULTS_DIR/${SUBJECT_ID}_ants_0GenericAffine.mat" \
        -v 2>> "$LOG_FILE"

    # Check if antsApplyTransforms succeeded
    if [ $? -ne 0 ]; then
        echo "Error during fMRI normalization for $SUBJECT_ID. Exiting." | tee -a "$LOG_FILE"
        return 1
    fi

    # Extract ROI time series from the normalized fMRI data
    echo "Extracting ROI time series for subject $SUBJECT_ID..." | tee -a "$LOG_FILE"
    fslmeants -i "$OUTPUT_DIR/${SUBJECT_ID}_fMRI_normalized.nii.gz" \
              -m "$ATLAS_ROI_PATH" -o "$OUTPUT_DIR/${SUBJECT_ID}_roi_timeseries.txt" \
              || { echo "Error during ROI extraction for $SUBJECT_ID" | tee -a "$LOG_FILE"; return 1; }

    echo "ROI extraction complete for subject $SUBJECT_ID at $(date)" | tee -a "$LOG_FILE"
}

# Process all subjects or a specific subject
if [ "$SUBJECT_ID" == "all" ]; then
    for SUBJECT_DIR in "$BASE_DIR"/preprocessing_results/*; do
        if [ -d "$SUBJECT_DIR" ]; then
            CURRENT_SUBJECT_ID=$(basename "$SUBJECT_DIR")
            normalize_and_extract_roi "$CURRENT_SUBJECT_ID" || {
                echo "Error processing subject $CURRENT_SUBJECT_ID"
                exit 1
            }
        fi
    done
else
    # Process a single subject
    normalize_and_extract_roi "$SUBJECT_ID"
fi
