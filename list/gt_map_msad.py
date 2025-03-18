import numpy as np
import pandas as pd
import os

# Paths to your files
feature_list_path = r'C:\Users\gteja\Documents\Python\VadCLIP\VadCLIP-MSAD\list\MSAD_train_list.list'  # Text file with feature paths
annotations_csv_path = r'C:\Users\gteja\Documents\Python\VadCLIP\VadCLIP-MSAD\list\anomaly_annotation.csv'  # Your annotation CSV
not_found = 0
# Load annotations
annotations_df = pd.read_csv(annotations_csv_path)

# Load feature file paths
with open(feature_list_path, 'r') as f:
    feature_paths = [line.strip() for line in f.readlines()]

gt_segment = []
gt_label = []

for path in feature_paths:
    # Extract video name from the path (e.g., "Assault_14" from "Assault_14_i3d.npy")
    filename = os.path.basename(path)
    video_name = filename.split('_i3d.npy')[0]
    #print(video_name)
    video_name = video_name.replace("MSAD_","")
    
    annotation_row = annotations_df[annotations_df['name'] == video_name]
    segment = []
    label = []
    
    if not annotation_row.empty:
        scenario = annotation_row['scenario'].values[0]
        start_frame = int(annotation_row['starting frame of anomaly'].values[0])
        end_frame = int(annotation_row['ending frame of anomaly'].values[0])
        # Add the anomaly segment
        segment.append([start_frame, end_frame])
        if 'normal' not in video_name:
            label.append('Anomaly')  # Using scenario as the anomaly label
    else:
        # If no annotation found (this could happen if annotation is missing for some videos)
        print(f"Warning: No annotation found for {video_name}")
        not_found += 1
        try:
            # Try to load the feature file to get its length
            # fea = np.load(path)
            # lens = fea.shape[0] * clip_len
            segment.append([-1, -1])
            label.append("Normal")  
        except Exception as e:
            print(f"Error loading feature file for {video_name}: {e}")
            
            segment.append([-1, -1])
            label.append("Error")
    
    
    gt_segment.append(segment)
    gt_label.append(label)

np.save('msad_gt_label_custom.npy', gt_label)
np.save('msad_gt_segment_custom.npy', gt_segment)

print(not_found)
print(f"Processed {len(feature_paths)} videos")
print(f"Labels and segments saved to list/gt_label_custom.npy and list/gt_segment_custom.npy")