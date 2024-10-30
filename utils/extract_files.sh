# Set the directory path
directory="/Users/muhammadmahajna/workspace/research/data/CVR/"

# Loop through all .tar.gz files in the specified directory
for file in "$directory"/*.tar.gz; do
    # Check if there are any .tar.gz files
    if [ -e "$file" ]; then
        # Remove the .tar.gz extension to create the folder name
        folder_name="${file%.tar.gz}"
        
        # Print the current file being processed
        echo "Processing $file..."
        
        # Create a directory with the name of the file (without .tar.gz)
        mkdir -p "$folder_name"
        
        # Extract the contents of the .tar.gz file into the newly created folder
        echo "Extracting $file into $folder_name/"
        tar -xzf "$file" -C "$folder_name"
        
        # Confirm extraction completion for the file
        echo "Extraction of $file completed!"
    else
        echo "No .tar.gz files found in $directory"
        break
    fi
done
