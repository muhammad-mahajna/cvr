#!/bin/bash
# (Formerly testscript.sh)


# Configurable base directory
DESIGN_DIR="../designs"

# Loop through subject IDs and process
for id in $(seq -w 1 26); do 
    subject="sub-$id"
    echo "Starting processing for $subject"
    
    cd "$subject"

    # Skull stripping if necessary
    if [[ ! -f "anat/${subject}_T1w_brain_f02.nii.gz" ]]; then 
        bet2 "anat/${subject}_T1w.nii.gz" "anat/${subject}_T1w_brain_f02.nii.gz" -f 0.2
    fi

    # Copy design files for feat analysis
    cp "$DESIGN_DIR/design_run1.fsf" .

    # Update subject ID in design file
    sed -i '' "s|sub-01|${subject}|g" design_run1.fsf

    # Run feat
    echo "Running feat for $subject"
    feat design_run1.fsf

    cd ..
done
