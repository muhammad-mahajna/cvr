#!/bin/bash

# Example Usage:
# ./preprocess_fmri_anatomical.sh /path/to/input /path/to/output SUBJECT001 SUBJECT002
# ./preprocess_fmri_anatomical.sh /path/to/input /path/to/output all

# Check if the correct number of arguments is provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <input_base_dir> <output_base_dir> <subject_id(s)> or 'all' to process all subjects"
    exit 1
fi

# Configurable directories
IN_BASE_DIR="$1"         # Directory where NIfTI files are stored (rsBOLD and T1)
OUTPUT_BASE_DIR="$2"     # Directory where processed data will be saved

# Shift arguments so that the third argument becomes the first subject ID
shift 2

# Function to preprocess a single subject
preprocess_subject() {
    SUBJECT_ID="$1"
    INPUT_RS_BOLD="$IN_BASE_DIR/${SUBJECT_ID}/rsBOLD_ET/${SUBJECT_ID}_rsBOLD_ET.nii.gz"
    INPUT_RS_BOLD_Flip="$IN_BASE_DIR/${SUBJECT_ID}/rsBOLD_ET_Flip/${SUBJECT_ID}_rsBOLD_ET_Flip.nii.gz"
    INPUT_T1="$IN_BASE_DIR/${SUBJECT_ID}/T1/${SUBJECT_ID}_T1.nii.gz"
    
    OUTPUT_DIR="$OUTPUT_BASE_DIR/preprocessing_results/$SUBJECT_ID"

    # Check if required NIfTI files exist
    if [ ! -f "$INPUT_RS_BOLD" ]; then
        echo "rsBOLD file not found: $INPUT_RS_BOLD"
        return 1
    fi
    if [ ! -f "$INPUT_T1" ]; then
        echo "T1 file not found: $INPUT_T1"
        return 1
    fi

    echo "Processing subject: $SUBJECT_ID"

    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"

    # Skull strip the T1-weighted image
    echo "Performing skull stripping on T1-weighted image for subject $SUBJECT_ID..."
    bet2 "$INPUT_T1" "$OUTPUT_DIR/${SUBJECT_ID}_strip" -f 0.65 || { echo "Error during skull stripping for $SUBJECT_ID"; return 1; }

    # Perform motion correction on rsBOLD images
    echo "Performing motion correction on fMRI images for subject $SUBJECT_ID..."
    mcflirt -in "$INPUT_RS_BOLD" -out "$OUTPUT_DIR/rsBOLD_motion_corrected" || { echo "Error during motion correction for $SUBJECT_ID"; return 1; }
    mcflirt -in "$INPUT_RS_BOLD_Flip" -out "$OUTPUT_DIR/rsBOLD_FLIP_motion_corrected" || { echo "Error during motion correction for $SUBJECT_ID"; return 1; }

    # Create acquisition parameters file for topup correction
    echo "Creating acquisition parameters file for subject $SUBJECT_ID..."
    acqparams_file="$OUTPUT_DIR/acqparams.txt"
    rm -f "$acqparams_file"  # Ensure the file is empty

    # Extract phase encoding direction and readout time from JSON files in rsBOLD_ET and rsBOLD_ET_Flip directories
    for file in "$IN_BASE_DIR/$SUBJECT_ID/rsBOLD_ET/"*.json "$IN_BASE_DIR/$SUBJECT_ID/rsBOLD_ET_Flip/"*.json; do
        if [ -f "$file" ]; then
            phase_encoding=$(jq -r '.PhaseEncodingDirection' "$file")
            readout_time=$(jq -r '.TotalReadoutTime' "$file")

            # Set phase vector based on encoding direction
            if [[ "$phase_encoding" == "j-" ]]; then
                phase_vector="0 -1 0"
            else
                phase_vector="0 1 0"
            fi

            echo "$phase_vector $readout_time" >> "$acqparams_file"
        fi
    done

    # First frame extraction for topup
    echo "Extracting first frame from motion-corrected fMRI data for subject $SUBJECT_ID..."
    fslroi "$OUTPUT_DIR/rsBOLD_motion_corrected" "$OUTPUT_DIR/rsBOLD_first_frame" 0 1 || { echo "Error extracting first frame for $SUBJECT_ID"; return 1; }
    fslroi "$OUTPUT_DIR/rsBOLD_FLIP_motion_corrected" "$OUTPUT_DIR/rsBOLD_FLIP_first_frame" 0 1 || { echo "Error extracting first frame for $SUBJECT_ID"; return 1; }

    # Merge echo images (optional if FLIP data is available)
    echo "Merging echo images for subject $SUBJECT_ID..."
    fslmerge -t "$OUTPUT_DIR/merged_epi" "$OUTPUT_DIR/rsBOLD_first_frame" "$OUTPUT_DIR/rsBOLD_FLIP_first_frame" || { echo "Error merging echo images for $SUBJECT_ID"; return 1; }

    # Apply topup correction
    echo "Applying topup correction for subject $SUBJECT_ID..."
    topup --imain="$OUTPUT_DIR/merged_epi" --datain="$acqparams_file" --config=b02b0.cnf --out="$OUTPUT_DIR/topup_results" || { echo "Error during topup correction for $SUBJECT_ID"; return 1; }

    echo "Applying topup results for subject $SUBJECT_ID..."
    applytopup --imain="$OUTPUT_DIR/rsBOLD_motion_corrected" --inindex=1 --topup="$OUTPUT_DIR/topup_results" --datain="$acqparams_file" --method=jac --out="$OUTPUT_DIR/rsBOLD_field_corrected" || { echo "Error applying topup results for $SUBJECT_ID"; return 1; }

    echo "Preprocessing complete for $SUBJECT_ID."
}

# If the argument is 'all', process all subjects in the input directory
if [ "$1" == "all" ]; then
    for SUBJECT_ID in $(ls "$IN_BASE_DIR"); do
        preprocess_subject "$SUBJECT_ID"
    done
else
    # Process each subject passed as an argument
    for SUBJECT_ID in "$@"; do
        preprocess_subject "$SUBJECT_ID"
    done
fi