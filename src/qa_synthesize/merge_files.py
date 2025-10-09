import os
import json

# Configuration - replace with your actual paths
output_file = 'YOUR_OUTPUT_PATH/hq_Electrical_Computer_and_Telecom_Engineering_synthesized_qa.json'  # e.g., "/path/to/output"

# List of JSON file paths
json_files = [
    'YOUR_QA_DATA_PATH/synthesized_qa/hq_Corporate_Finance_and_Treasury_synthesized_qa_template_img_only_2_remained_filtered_rewritten.json',
    'YOUR_QA_DATA_PATH/synthesized_qa/hq_Corporate_Finance_and_Treasury_synthesized_qa_template_remained_filtered_merged_rewritten.json',
    # Add more file paths as needed
]

# List to hold all dictionaries
merged_data = []

# Iterate over the provided file paths
data = []
for file_path in json_files:
    with open(file_path, 'r') as file:
        data += list(file)

# Write the merged data to the output file
with open(output_file, 'w') as file:
    for element in data:
        file.write(element)
        file.flush()

# Merged {len(data)} entries into output file