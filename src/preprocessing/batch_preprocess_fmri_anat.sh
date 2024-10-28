#!/bin/bash

# Example Usage:
# ./batch_preprocess_fmri_anat.sh /data/project1 SUBJECT001 SUBJECT002
# ./batch_preprocess_fmri_anat.sh /data/project1 all

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <base_directory> <subject_id(s)> or 'all' to process all subjects"
    exit 1
fi

# Configurable base directory
BASE_DIR="$1"  # First argument is the base directory
INPUT_BASE_DIR="$BASE_DIR/RAW_DATA/undone"
OUTPUT_BASE_DIR="$BASE_DIR/RAW_DATA/completed"

# Shift arguments so that $2 becomes the first subject ID
shift

# Function to preprocess a single subject
preprocess_subject() {
    SUBJECT_ID="$1"
    INPUT_DIR="$INPUT_BASE_DIR/$SUBJECT_ID"
    OUTPUT_DIR="$OUTPUT_BASE_DIR/$SUBJECT_ID"

    # Check if the subject directory exists
    if [ ! -d "$INPUT_DIR" ]; then
        echo "Directory not found: $INPUT_DIR"
        return 1
    fi

    # Navigate to the subject's data directory
    cd "$INPUT_DIR" || return 1

    echo "Processing directory: $INPUT_DIR"

    # Create necessary directories for images and processing if they don't exist
    mkdir -p images processing

    # Convert DICOM files to NIfTI format
    echo "Converting DICOM to NIfTI for subject $SUBJECT_ID..."
    dcm2niix -f "${SUBJECT_ID}_rsBOLD_ET" -o ./images/ ./*-rsBOLD-End-tidal || { echo "Error converting rsBOLD files for $SUBJECT_ID"; return 1; }
    dcm2niix -f "${SUBJECT_ID}_T1" -o ./images/ ./*-SAG_FSPGR_BRAVO || { echo "Error converting T1 files for $SUBJECT_ID"; return 1; }

    # Skull strip the T1-weighted image
    echo "Performing skull stripping on T1-weighted image for subject $SUBJECT_ID..."
    bet2 ./images/*T1.nii ./processing/${SUBJECT_ID}_brain -f 0.65 || { echo "Error during skull stripping for $SUBJECT_ID"; return 1; }

    # Perform motion correction on rsBOLD images
    echo "Performing motion correction on fMRI images for subject $SUBJECT_ID..."
    mcflirt -in ./images/*rsBOLD_ET.nii -out ./processing/rsBOLD_mc || { echo "Error during motion correction for $SUBJECT_ID"; return 1; }

    # Create acquisition parameters file for topup correction
    echo "Creating acquisition parameters file for subject $SUBJECT_ID..."
    touch acqparams.txt
    for file in $(ls ./images/*rsBOLD*.json); do
        phase_encoding=$(jq -r '.PhaseEncodingDirection' "$file")
        readout_time=$(jq -r '.TotalReadoutTime' "$file")

        # Set phase vector based on encoding direction
        if [[ "$phase_encoding" == "j-" ]]; then
            phase_vector="0 -1 0"
        else
            phase_vector="0 1 0"
        fi

        echo "$phase_vector $readout_time" >> acqparams.txt
    done

    # Move acquisition parameters file to processing directory
    mv acqparams.txt ./processing

    # First frame extraction for topup
    echo "Extracting first frame from motion-corrected fMRI data for subject $SUBJECT_ID..."
    fslroi ./processing/rsBOLD_mc ./processing/rsBOLD_ff 0 1 || { echo "Error extracting first frame for $SUBJECT_ID"; return 1; }

    # Merge echo images (optional if FLIP data is available)
    echo "Merging echo images for subject $SUBJECT_ID..."
    fslmerge -t merged_epi ./processing/rsBOLD_ff || { echo "Error merging echo images for $SUBJECT_ID"; return 1; }

    # Apply topup correction
    echo "Applying topup correction for subject $SUBJECT_ID..."
    topup --imain=merged_epi --datain=./processing/acqparams.txt --config=b02b0.cnf --out=topup_results || { echo "Error during topup correction for $SUBJECT_ID"; return 1; }

    echo "Applying topup results for subject $SUBJECT_ID..."
    applytopup --imain=./processing/rsBOLD_mc --inindex=1 --topup=topup_results --datain=./processing/acqparams.txt --method=jac --out=./processing/rsBOLD_fc || { echo "Error applying topup results for $SUBJECT_ID"; return 1; }

    echo "Preprocessing complete for $SUBJECT_ID."

    # Move the processed data to the completed folder
    mv "$INPUT_DIR" "$OUTPUT_DIR" || { echo "Error moving directory to completed folder for $SUBJECT_ID"; return 1; }

    echo "Data moved to $OUTPUT_DIR"
}

# If the argument is 'all', process all subjects in the input directory
if [ "$1" == "all" ]; then
    for SUBJECT_ID in $(ls "$INPUT_BASE_DIR"); do
        preprocess_subject "$SUBJECT_ID"
    done
else
    # Process each subject passed as an argument
    for SUBJECT_ID in "$@"; do
        preprocess_subject "$SUBJECT_ID"
    done
fi

