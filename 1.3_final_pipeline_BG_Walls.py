import os
import glob
import cv2
import numpy as np
import yaml
from dataclasses import asdict

from utils.logger_ import log_
from key_functions import get_illuminant_from, to_float64, to_uint8, check_memory, free_memory
from ColorCorrection import ColorCorrection
from kwargs import Config

import gc, time
from copy import deepcopy

gc.enable()

#---------------------------------------Configurations-------------------------------------------------------    
DATA_FOLDER = 'Data'
TEST_GROUP = 'Walls' # 'Backgrounds' or 'Walls'

FO_ = {
    'Black_D65',
    'White_D65',
}

# FO_ = {
#     'Black',
#     'White',
#     'Colors',
# }

ILLUMINANT_FROM = 'D65'
CMFS = "CIE 1931 2 Degree Standard Observer"

SEQUENCES_ = [
    [False, True, True, True, True],
]

# run for last 2 sequences (currently using all sequences)
SEQUENCES = SEQUENCES_

# Degrees = [1, 2, 3, 4, 5]
Degrees = [2]

color_method = 'ours'  # 'ours' or 'conv'

# run_pred, show_pred, save
run_predict, show_predict, save = True, False, True
Name_Prefix = ""

Blank_Image_Name = 'Blank'
Image_for_Color_Correction = 'CC'
Other_Images_name = 'Sample'


# Initialize config
g_config = Config()

folder = os.path.join(DATA_FOLDER, TEST_GROUP)
folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f)) and (f in FO_)]

a=1

#---------------------------------------Functions-------------------------------------------------------

def run_color_correction(image, White_Image=None, save_path=None, config=None):
    check_memory()
    basename = os.path.basename(image).split('.')[0]
    log_(f'Running CC on "{basename}"...', 'cyan', 'italic')
    CC_instance = ColorCorrection()
    
    image_bgr = cv2.imread(image)
    if image_bgr is None:
        log_(f'Image "{basename}" not found', 'red', 'italic', 'warn')
        return None, None

    image_rgb = to_float64(image_bgr[:, :, ::-1])
    # Read the white image if provided
    white_img = cv2.imread(White_Image) if White_Image else None
    ALL_METRICS, IMAGES, _ = CC_instance.Run_(Image=image_rgb, White_Image=white_img, name_=basename, config=config)

    # get delta E from ALL_METRICS
    DE = ALL_METRICS[f"{basename}_CC"]['Summary']['DE_mean'].values[-1]
    if save:
        for k, v in IMAGES.items():
            cv2.imwrite(os.path.join(save_path, f'{k}.jpg'), to_uint8(v[:, :, ::-1]))

    del IMAGES, ALL_METRICS, image_bgr, image_rgb
    free_memory()
    return CC_instance, DE

def run_inference(image, CC_pred_local, save_path=None):
    if CC_pred_local is None:
        log_(f'CC_pred is None, skipping inference for image {os.path.basename(image)}', 'red', 'italic', 'warn')
        return
    check_memory()
    image_name = os.path.basename(image).split('.')[0]
    log_(f'Predicting "{image_name}"...', 'cyan', 'italic')
    
    image_bgr = cv2.imread(image)
    if image_bgr is None:
        log_(f'Image "{image_name}" not found', 'red', 'italic', 'warn')
        return

    image_rgb = to_float64(image_bgr[:, :, ::-1])
    Image_corrected = CC_pred_local.Predict_Image(Image=image_rgb, show=show_predict)
    
    if save:
        out_path = os.path.join(save_path, f'{image_name}_Corrected.jpg')
        cv2.imwrite(out_path, to_uint8(Image_corrected['CC'][:, :, ::-1]))

    del image_bgr, image_rgb, Image_corrected
    free_memory()

def process_folder(f_):
    # Create a deep copy of the global configuration for this folder
    config = deepcopy(g_config)
    
    matched_folder = os.path.basename(f_)

    current_folder = os.path.join(folder, matched_folder)
    log_(f'Processing Folder: "{matched_folder}"...', 'magenta', 'italic')

    mtd = color_method
    if color_method == 'ours':
        mtd = config.CC_kwargs.mtd

    suffix = f'_{config.CC_kwargs.degree}Deg_{mtd}'
    save_path = os.path.join(folder, 'Results', Name_Prefix + 'Results' + suffix, matched_folder)

    if save and not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    config.save_path = save_path

    REF_ILLUMINANT = get_illuminant_from(ILLUMINANT_FROM, CMFS=CMFS)
    config.REF_ILLUMINANT = REF_ILLUMINANT

    image_files = glob.glob(os.path.join(current_folder, '*.jpg'))
    image_basenames = [os.path.basename(fp) for fp in image_files]

    id_blank = next((i for i, b in enumerate(image_basenames) if Blank_Image_Name.lower() in b.lower().split('_')[0]), None)
    ids_CC = [i for i, b in enumerate(image_basenames) if Image_for_Color_Correction.lower() in b.lower().split('_')[0]]
    ids_Other = [i for i, b in enumerate(image_basenames) if Other_Images_name.lower() in b.lower().split('_')[0]]
    
    White_Image = image_files[id_blank] if id_blank is not None else None
    CC_Images = [image_files[i] for i in ids_CC] if ids_CC else []
    Other_Images = [image_files[i] for i in ids_Other] if ids_Other else []

    # Run color correction on available images
    CC_pred = None
    if CC_Images:
        CC_good = []
        mean_DEs = []
        for i, image in enumerate(CC_Images):
            basename = os.path.basename(image).split('.')[0]
            log_(f'Running color correction on Image {i+1}/{len(CC_Images)}... "{basename}"...', 'cyan', 'italic')
            try:
                cc_instance, DE_ = run_color_correction(
                    image=image, 
                    White_Image=White_Image, 
                    save_path=save_path, 
                    config=config,
                )
            except Exception as e:
                log_(f'Error running color correction on Image {i+1}/{len(CC_Images)}... "{basename}"...', 'red', 'italic', 'warn')
                log_(f'{e}', 'red', 'italic', 'warn')
                continue
            
            if cc_instance is not None:
                CC_good.append(cc_instance)
                mean_DEs.append(DE_)
        
        if CC_good:
            idx_min = np.argmin(np.array(mean_DEs))
            CC_pred = CC_good[idx_min]
        else:
            log_(f'No valid color correction results for folder "{matched_folder}".', 'magenta', 'italic')
    else:
        log_(f'No color correction images found in folder "{matched_folder}".', 'magenta', 'italic')
    
    Other_Images_ = []
    Other_Images_ = list(set(Other_Images+CC_Images))
    log_(f'Other_Images: \n\n{Other_Images_}', 'green', 'italic')
    # Run inference only if a valid CC prediction exists and there are images to process
    if run_predict and Other_Images_:
        if CC_pred is not None:
            log_(f'Running inference on {len(Other_Images_)} images...', 'cyan', 'italic')
            for image in Other_Images_:
                run_inference(image, CC_pred, save_path=save_path)
        else:
            log_(f'Skipping inference because no valid CC prediction is available for folder "{matched_folder}".', 'magenta', 'italic')
    else:
        log_(f'Skipping inference for folder "{matched_folder}".', 'magenta', 'italic')

    if save:
        config_path = os.path.join(save_path, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
        log_(f'Saved config to "{config_path}"', 'green', 'italic')

def main():
    global g_config, Name_Prefix

    for seq_ in SEQUENCES:
        DO_FFC, DO_GC, DO_WB, DO_CC, CHECK_SAT = seq_
        Name_Prefix = f"With_{'FFC_' * DO_FFC}{'GC_' * DO_GC}{'WB_' * DO_WB}{'CC_' * DO_CC}"

        log_(f'Processing Sequence: {Name_Prefix}...', 'magenta', 'Bold')

        # Update global configuration
        g_config.do_ffc, g_config.do_gc, g_config.do_wb, g_config.do_cc = DO_FFC, DO_GC, DO_WB, DO_CC
        g_config.Saturation_kwargs.check_saturation = CHECK_SAT
        g_config.cc_method = color_method
        g_config.update()

        for deg_ in Degrees:
            if deg_ == 5 and color_method == 'conv':
                log_(f'Skipping Degree: {deg_} for method: {color_method}...', 'magenta', 'italic')
                continue
            g_config.CC_kwargs.degree = deg_
            log_(f'Processing for Degree: {deg_}...', 'magenta', 'italic')
            
            for f in folders:
                process_folder(f)

    log_(f'Done...', 'green', 'Bold', 'info')

if __name__ == '__main__':  
    gc.enable()
    tic = time.time()
    main()
    toc = time.time()
    log_(f'Done in {toc - tic:.2f} seconds', 'green', 'italic')
    free_memory()
