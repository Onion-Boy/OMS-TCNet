from light_training.preprocessing.preprocessors.preprocessor_mri import MultiModalityPreprocessor 
import numpy as np 
import pickle 
import json 

data_filename = "LISA_ciso.nii.gz",
seg_filename = ""

def process_train():
    base_dir = "/media/ly/LISA_Task2b"
    image_dir = "baga_validation"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )

    out_spacing = [1.0, 1.0, 1.0]
    output_dir = "/media/ly/LISA_Task2b/validation_fullres_process/"
    
    preprocessor.run(output_spacing=out_spacing, 
                     output_dir=output_dir, 
                     all_labels=[1, 2, 3, 4],
    )

def plan():
    base_dir = "/media/ly/LISA_Task2b"
    image_dir = "baga_validation"
    preprocessor = MultiModalityPreprocessor(base_dir=base_dir, 
                                    image_dir=image_dir,
                                    data_filenames=data_filename,
                                    seg_filename=seg_filename
                                   )
    analysis_path = "/media/ly/LISA_Task2b/baga_validation/data_analysis_result.txt"
    preprocessor.run_plan(analysis_path)


if __name__ == "__main__":
# 
    # plan()
    process_train()
  
