import splitfolders

splitfolders.ratio('F:/Data_for_Deepfake_DL/Celeb-DF-v2/', 
                   output="F:/Data_for_Deepfake_DL/Dataset-1/",
                   ratio=(0.8,0.0,0.2)) # 80% training, 0% validation, 20% testing
                   