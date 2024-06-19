import splitfolders

splitfolders.ratio('F:/Data_for_Deepfake_DL/Dataset-1/train', 
                   output="F:/Data_for_Deepfake_DL/Dataset-1/new-train",
                   ratio=(0.8,0.2)) # 80% training, 20% validation
                   