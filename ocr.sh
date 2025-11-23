#!/bin/bash
# Script to run Tesseract on all JPG files in the /app/frames directory

# The -L eng flag specifies the English language for OCR
for img_file in /app/frames/*.jpg; do
    # Check if the file exists (handles the case where no jpgs are found)
    echo ${img_file}
    if [ -f "$img_file" ]; then
        # Tesseract syntax: tesseract <input> <output_base_name>
        # Output will be <filename>.txt
        base_name=$(basename "$img_file" .jpg)
        tesseract "$img_file" "/app/frames/$base_name" -l eng 
        echo "Processed: $img_file"
    fi
done
echo "OCR processing complete."
