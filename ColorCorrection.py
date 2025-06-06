#------------------------------------------------------------------------------------------------------------------------------
# 1. Imports and Constants                                                                                                      
#------------------------------------------------------------------------------------------------------------------------------

import colour.plotting
import cv2
import colour
import numpy as np
import pandas as pd
import pickle as pkl
import os, time  
import torch
import gc

from utils.logger_ import log_
from FFC.FF_correction import FlatFieldCorrection
from key_functions import *

gc.enable()


def get_attr(obj, attr, default=None):
    return getattr(obj, attr, default) if obj is not None else default

class MyModels:
    def __init__(self):
        self.model_ffc = None # stores the FFC multiplier
        self.model_cc = None # stores the color correction model or matrix
        self.model_wb = None # stores the white balance model diagonal matrix
        self.model_gc = None # stores the gamma correction fit coefficients

    def save_models(self, path, name_=''):
        # save self as a pickle file
        name_ = os.path.join(path, name_+'_models.pkl')
        with open(name_, 'wb') as f:
            pkl.dump(self, f)


    def load_models(self, path, name_=''):

        with open(os.path.join(path, name_+'_models.pkl'), 'rb') as f:
            self_ = pkl.load(f)

        self.model_ffc = self_.model_ffc # stores the FFC multiplier
        self.model_cc = self_.model_cc # stores the color correction model or matrix
        self.model_wb = self_.model_wb # stores the white balance model diagonal matrix
        self.model_gc = self_.model_gc # stores the gamma correction fit coefficients


class ColorCorrection:
    def __init__(self):
        self.Image = None
        self.White_Image = None
        self.Models = MyModels()
        self.Models_path = None
        self.Results = None

        self.is_saturated = False

        self.REFERENCE_CHART = None
        self.REF_ILLUMINANT = None
        self.REFERENCE_RGB_PD = None
        self.REFERENCE_NEUTRAL_PATCHES_PD = None
        
        
    def do_flat_field_correction(self, Image=None, do_ffc=True, ffc_kwargs=None):
        
        if Image is None:
            Image = self.Image
        try:
            if (self.White_Image is not None) and do_ffc:
                # white image is already a bgr 8bit
                # convert the image from 0-1 64 bit float RGB to 0-255 8 bit unsigned int BGR
                Image = to_uint8(Image[:, :, ::-1])
                assert self.White_Image.shape == Image.shape, 'Image and white image must have the same shape'
                get_deltaE = get_attr(ffc_kwargs, 'get_deltaE', True)
                ffc_kwargs_ = {
                    'model_path': get_attr(ffc_kwargs, 'model_path', ''),
                    'manual_crop': get_attr(ffc_kwargs, 'manual_crop', True if get_attr(ffc_kwargs, 'model_path', '') == '' else False),
                    'show': get_attr(ffc_kwargs, 'show', False),
                    'bins': get_attr(ffc_kwargs, 'bins', 50),
                    'smooth_window': get_attr(ffc_kwargs, 'smooth_window', 5),
                    'crop_rect': get_attr(ffc_kwargs, 'crop_rect', None),
                }
                fit_kwargs = {
                    'degree': get_attr(ffc_kwargs, 'degree', 3),
                    'interactions': get_attr(ffc_kwargs, 'interactions', False),
                    'fit_method': get_attr(ffc_kwargs, 'fit_method', 'linear'),
                    'max_iter': get_attr(ffc_kwargs, 'max_iter', 1000),
                    'tol': get_attr(ffc_kwargs, 'tol', 1e-8),
                    'verbose': get_attr(ffc_kwargs, 'verbose', False),
                    'random_seed': get_attr(ffc_kwargs, 'random_seed', 0),
                }


                ffc = FlatFieldCorrection(self.White_Image, **ffc_kwargs_)
                multiplier = ffc.compute_multiplier(**fit_kwargs)

                c_Image = ffc.apply_ffc(
                    Image, 
                    multiplier, 
                    show=get_attr(ffc_kwargs, 'show', False)
                )

                Metrics_ = {}
                if get_deltaE:

                    ref_ = self.REFERENCE_RGB_PD.values
                    illum = self.REF_ILLUMINANT
                    # get deltaE for chart in image before and after ffc

                    _, cps_before = extract_neutral_patches(Image, return_one=True)
                    _, cps_after = extract_neutral_patches(c_Image, return_one=True)

                    metrics_before = get_metrics(ref_, cps_before.values, illum, 'srgb')
                    metrics_after = get_metrics(ref_, cps_after.values, illum, 'srgb')

                    Metrics_ = arrange_metrics(metrics_before, metrics_after, name='FFC')


                c_Image = to_float64(c_Image[:, :, ::-1])
                
                self.Models.model_ffc = multiplier

                return c_Image, Metrics_, False

            else:
                log_("skipping flat field correction", 'light_yellow', 'italic', 'warning')
                log_("white image is None", 'light_yellow', 'italic', 'warning') if self.White_Image is None else None
                log_("FFC mode is disabled", 'light_yellow', 'italic', 'warning') if do_ffc == False else None
                return Image, None, False
        
        except Exception as e:
            log_(e, 'red', 'bold')
            return Image, None, True

    def check_saturation(self, Image=None, do_check=True):
        try:
            if do_check:
                if Image is None:
                    Image = self.Image

                Image, values, ids = extrapolate_if_sat_image(Image, self.REFERENCE_RGB_PD.values)
                return Image, values, ids
            else:
                log_("skipping saturation check", 'light_yellow', 'italic', 'warning')

                return Image
            
        except Exception as e:
            log_(e, 'red', 'bold')

            return Image


    def do_gamma_correction(self, Image=None, do_gc=True, gc_kwargs=None):

        if Image is None:
            Image = self.Image

        try:
            if do_gc:

                gc_kwargs_dict = {
                    'max_degree': get_attr(gc_kwargs, 'max_degree', 5),
                    'show': get_attr(gc_kwargs, 'show', False),
                    'get_deltaE': get_attr(gc_kwargs, 'get_deltaE', True),
                }

                coeffs_gc, img_gc, Metrics_gc = estimate_gamma_profile(
                    img_rgb=Image,
                    ref_cp=self.REFERENCE_RGB_PD.values,
                    ref_illuminant=self.REF_ILLUMINANT,
                    **gc_kwargs_dict,
                )

                self.Models.model_gc = coeffs_gc

                return np.clip(img_gc, 0, 1), Metrics_gc, False

            else:
                log_("skipping gamma correction", 'light_yellow', 'italic', 'warning')
                log_("gamma correction mode is disabled", 'light_yellow', 'italic', 'warning') if do_gc == False else None

                return Image, None, False
            
        except Exception as e:
            log_(e, 'red', 'bold')
            return Image, None, True
            

    def do_white_balance(self, Image=None, do_wb=True, wb_kwargs=None):
        
        if Image is None:
            Image = self.Image

        try:
            if do_wb:

                wb_kwargs_dict = {
                    'show': get_attr(wb_kwargs, 'show', False),
                    'get_deltaE': get_attr(wb_kwargs, 'get_deltaE', True),
                }

                diag_wb, img_wb, Metrics_wb = wb_correction(
                    img_rgb=Image,
                    ref_cp=self.REFERENCE_RGB_PD.values,
                    ref_illuminant=self.REF_ILLUMINANT,
                    **wb_kwargs_dict,
                )

                self.Models.model_wb = diag_wb

                return np.clip(img_wb, 0, 1), Metrics_wb, False

            else:
                log_("skipping white balance", 'light_yellow', 'italic', 'warning')
                log_("white balance mode is disabled", 'light_yellow', 'italic', 'warning') if do_wb == False else None

                return Image, None, False
            
        except Exception as e:
            log_(e, 'red', 'bold')
            return Image, None, True


    def do_color_correction(self, Image=None, do_cc=True, cc_method='ours', cck=None):

        if Image is None:
            Image = self.Image
        
        try:
            if do_cc:

                if cc_method == 'conv':
                    cc_kwargs_dict = {
                        'method': get_attr(cck, 'method', "Finlayson 2015"),
                        'degree': get_attr(cck, 'degree', 3),
                        'root_polynomial_expansion': get_attr(cck, 'root_polynomial_expansion', None),
                        'terms': get_attr(cck, 'terms', None),
                    }

                    log_(f"Color correction method: '{cc_kwargs_dict['method']}'", 'cyan', 'italic')

                    ccm1, img_cc1, corrected_card, Metrics_cc1 = color_correction_1(
                        img_rgb=Image,
                        ref_rgb=self.REFERENCE_RGB_PD.values,
                        ref_illuminant=self.REF_ILLUMINANT,
                        show = get_attr(cck, 'show', False),
                        get_deltaE = get_attr(cck, 'get_deltaE', True),
                        cc_kwargs = cc_kwargs_dict,
                    )

                    if get_attr(cck, 'show', False):
                        colour.plotting.plot_multi_colour_checkers([
                            self.REFERENCE_CHART,
                            corrected_card,
                        ])

                    self.Models.model_cc = [ccm1, cc_kwargs_dict, cc_method]

                    return np.clip(img_cc1, 0, 1), Metrics_cc1, False
                
                elif cc_method == 'ours':
                    cc_kwargs_dict = {
                        'mtd': get_attr(cck, 'mtd', 'linear'),
                        'degree': get_attr(cck, 'degree', 3),
                        'max_iterations': get_attr(cck, 'max_iterations', 1000),
                        'nlayers': get_attr(cck, 'nlayers', 100),
                        'ncomp': get_attr(cck, 'ncomp', -1),
                        'tol': get_attr(cck, 'tol', 1e-8),
                        'random_state': get_attr(cck, 'random_state', 0),
                        'verbose': get_attr(cck, 'verbose', False),
                        'param_search': get_attr(cck, 'param_search', False),
                        'hidden_layers': get_attr(cck, 'hidden_layers', [64, 32, 16]),
                        'learning_rate': get_attr(cck, 'learning_rate', 0.001),
                        'batch_size': get_attr(cck, 'batch_size', 32),
                        'patience': get_attr(cck, 'patience', 10),
                        'dropout_rate': get_attr(cck, 'dropout_rate', 0.1),
                        'use_batch_norm': get_attr(cck, 'use_batch_norm', False),
                        'optim_type': get_attr(cck, 'optim_type', 'Adam'),
                    }

                    log_(f"Color correction method: '{cc_kwargs_dict['mtd']}'", 'cyan', 'italic')

                    model, img_cc2, corrected_card, Metrics_cc2 = color_correction(
                        img_rgb=Image,
                        ref_rgb=self.REFERENCE_RGB_PD.values,
                        ref_illuminant=self.REF_ILLUMINANT,
                        show= get_attr(cck, 'show', False),
                        get_deltaE = get_attr(cck, 'get_deltaE', True),
                        cc_kwargs = cc_kwargs_dict,
                        n_samples=get_attr(cck, 'n_samples', 50),
                    )

                    if get_attr(cck, 'show', False):
                        colour.plotting.plot_multi_colour_checkers([
                            self.REFERENCE_CHART,
                            corrected_card, #corrected_card,
                        ])

                    self.Models.model_cc = [model, cc_kwargs_dict, cc_method]

                    return np.clip(img_cc2, 0, 1), Metrics_cc2, False

                else:
                    log_("Error: invalid color correction method", 'red', 'bold')
                    log_("Try 'convetional' or 'ours'", 'red', 'bold')

                    log_("skipping color correction", 'light_yellow', 'italic', 'warning')
                    log_("color correction mode is disabled", 'light_yellow', 'italic', 'warning') if do_cc == False else None

                    return Image, None, False

        except Exception as e:
            log_(e, 'red', 'bold')
            return Image, None, True
        

    def get_reference_values(self, REF_ILLUMINANT=None):

        if REF_ILLUMINANT is None:
            REF_ILLUMINANT = self.REF_ILLUMINANT

        REFERENCE_CHART = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']

        # Do chromatic adaptation from ICC D50 to detected illuminant (ICC D50 is the default and same as D50)
        REFERENCE_CHART = adapt_chart(REFERENCE_CHART, REF_ILLUMINANT)

        data = list(REFERENCE_CHART.data.values())
        names = list(REFERENCE_CHART.data.keys())

        REFERENCE_RGB = colour.XYZ_to_sRGB(
            colour.xyY_to_XYZ(data),    
            illuminant=REF_ILLUMINANT,
            apply_cctf_encoding=True
        )

        REFERENCE_NEUTRAL_PATCHES_PD = pd.DataFrame(
            REFERENCE_RGB[-6:],
            columns=['R', 'G', 'B'],
            index=names[-6:]
        )

        REFERENCE_RGB_PD = pd.DataFrame(
            np.clip(REFERENCE_RGB, 0, 1),
            # REFERENCE_RGB,
            columns=['R', 'G', 'B'],
            index=names
        )

        self.REFERENCE_CHART = REFERENCE_CHART
        self.REF_ILLUMINANT = REF_ILLUMINANT
        self.REFERENCE_RGB_PD = REFERENCE_RGB_PD
        self.REFERENCE_NEUTRAL_PATCHES_PD = REFERENCE_NEUTRAL_PATCHES_PD

        return REFERENCE_RGB_PD
    
    
    
    def Run_(self, 
             Image, # can be path to image or numpy array (RGB, 0-1, float64)
             White_Image, # can be path to image or numpy array (BGR, 0-255, uint8)
             name_='', # name to save image
             config=None, # is instance of Config with default values
             ):
        
        log_("Initializing... Color Correction pipeline".upper(), 'Light_blue', 'bold')


        # if Image is not path --> range 0 to 1, 64 bit float
        # if path --> read image, convert to 64 bit float
        # Image_bgr is 8 bit uint image
        
        if isinstance(Image, str):
            Image_bgr = cv2.imread(Image)
            self.Image = to_float64(Image_bgr[:, :, ::-1])
        elif isinstance(Image, np.ndarray):
            self.Image = Image

        if isinstance(White_Image, str):
            self.White_Image = cv2.imread(White_Image)
        elif isinstance(White_Image, np.ndarray):
            self.White_Image = White_Image

        self.White_Image = White_Image

        do_ffc = get_attr(config, 'do_ffc', True)
        do_gc = get_attr(config, 'do_gc', True)
        do_wb = get_attr(config, 'do_wb', True)
        do_cc = get_attr(config, 'do_cc', True)
        do_sat = get_attr(config.Saturation_kwargs, 'check_saturation', True)

        ffc_kwargs = get_attr(config, 'FFC_kwargs', {})
        gc_kwargs = get_attr(config, 'GC_kwargs', {})
        wb_kwargs = get_attr(config, 'WB_kwargs', {})
        cc_kwargs = get_attr(config, 'CC_kwargs', {})
        

        save_ = get_attr(config, 'save', False)
        REF_ILLUMINANT_ = get_attr(config, 'REF_ILLUMINANT', colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
        if save_:
            save_path = get_attr(config, 'save_path', None)

        IMAGES_ = {}

        # get reference values
        self.get_reference_values(
            REF_ILLUMINANT=REF_ILLUMINANT_
        )

        # print(self.REFERENCE_RGB_PD)

        # print(cc_kwargs)

        #########################################################################################################################################################

        log_("1. Doing Flat Field Correction...".upper(), 'Light_blue', 'bold') if do_ffc else None

        tic = time.time()
        # do flat field correction
        Image_ffc, Metrics_FFC, e_ffc = self.do_flat_field_correction(
            Image=self.Image,
            do_ffc=do_ffc,
            ffc_kwargs=ffc_kwargs,
        )
        
        elapsed = time.time() - tic
        log_(f"Flat field correction done in {elapsed:.2f} seconds", 'light_green', 'italic') if do_ffc else None


        #########################################################################################################################################################

        log_("2. Detecting saturated color patches...".upper(), 'Light_blue', 'bold') if do_sat else None
        tic = time.time()
        # do saturation correction if required
        if do_sat:
            Image_sat, values_, ids_ = self.check_saturation(
                Image=Image_ffc,
                do_check=do_sat,
            )

            if ids_ is not None:
                Sat_data = pd.DataFrame(
                    {
                        'Image': [name_]*len(ids_),
                        'ID': ids_,
                        'Value_R': values_[:,0],
                        'Value_G': values_[:,1],
                        'Value_B': values_[:,2],
                    }
                )

                # save to csv
                Sat_data.to_csv(os.path.join(save_path, f'{name_}_Sat_data.csv'), float_format='%.9f', encoding='utf-8-sig')
        else:
            Image_sat = Image_ffc
        elapsed = time.time() - tic
        log_(f"Saturation correction done in {elapsed:.2f} seconds", 'light_green', 'italic') if do_sat else None

        #########################################################################################################################################################

        log_("3. Estimating Gamma Correction...".upper(), 'Light_blue', 'bold')if do_gc else None
        tic = time.time()
        # do gamma correction
        Image_gc, Metrics_GC, e_gc = self.do_gamma_correction(
            Image=Image_sat,
            do_gc=do_gc,
            gc_kwargs=gc_kwargs,
        )
        elapsed = time.time() - tic
        log_(f"Gamma correction done in {elapsed:.2f} seconds", 'light_green', 'italic') if do_gc else None

        #########################################################################################################################################################

        log_("4. Doing White Balance...".upper(), 'Light_blue', 'bold') if do_wb else None
        tic = time.time()
        # do white balance
        Image_wb, Metrics_WB, e_wb = self.do_white_balance(
            Image=Image_gc,
            do_wb=do_wb,
            wb_kwargs=wb_kwargs,
        )
        elapsed = time.time() - tic
        log_(f"White balance done in {elapsed:.2f} seconds", 'light_green', 'italic') if do_wb else None

        #########################################################################################################################################################

        log_("5. Doing Color Correction...".upper(), 'Light_blue', 'bold') if do_cc else None
        tic = time.time()

        # do color correction
        Image_cc, Metrics_CC, e_cc = self.do_color_correction(
            Image=Image_wb,
            do_cc=do_cc,
            cc_method = get_attr(config, 'cc_method', 'ours'),
            cck=cc_kwargs,
        )

        elapsed = time.time() - tic
        log_(f"Color correction done in {elapsed:.2f} seconds", 'light_green', 'italic') if do_cc else None


        #########################################################################################################################################################

        Images__ = [Image_ffc, Image_gc, Image_wb, Image_cc]
        dos__ = [do_ffc, do_gc, do_wb, do_cc]
        Metrics__ = [Metrics_FFC, Metrics_GC, Metrics_WB, Metrics_CC]
        Suffix__ = ['FFC', 'GC', 'WB', 'CC']
        
        IMAGES_  = {}
        ALL_METRICS = {}
        for IM, metric, suffix, do_ in zip(Images__, Metrics__, Suffix__, dos__):
            if do_:
                IMAGES_[f'{name_}_{suffix}'] = IM
                ALL_METRICS[f'{name_}_{suffix}'] = metric

        if save_:
            # save the models to pickle file
            # self.Models.save_models(save_path, name_)
            # log_(f"Models for {name_} saved to {save_path}", 'light_green', 'italic', 'info')  

            row_names = list(self.REFERENCE_CHART.data.keys())
            # save the Metrics results as csv files
            METRICS = pd.DataFrame()
            SUMMARY = pd.DataFrame()
            for k, v in ALL_METRICS.items():
                try:
                    all_metrics = v['All']
                    summary_metrics = v['Summary']



                # horizontal concatenation
                    METRICS = pd.concat([METRICS, all_metrics], axis=1)

                # vertical concatenation
                    SUMMARY = pd.concat([SUMMARY, summary_metrics], axis=0)

                except:
                    pass

            # make the indeces of METRICS as row names
            METRICS.index = row_names

            # save the metrics to csv files
            METRICS.to_csv(os.path.join(save_path, f'{name_}_All_Metrics.csv'), float_format='%.9f', encoding='utf-8')
            SUMMARY.to_csv(os.path.join(save_path, f'{name_}_Summary_Metrics.csv'), float_format='%.9f', encoding='utf-8')

            log_(f"Metrics for '{name_}' saved to '{save_path}'", 'cyan', 'italic', 'info')

            log_(f'METRICS SUMMARY:\n{SUMMARY}', 'cyan', 'italic')

        Error_ = any([e_ffc, e_gc, e_wb, e_cc])

        # clear cuda resources if any
        torch.cuda.empty_cache()
        gc.collect()

        return  ALL_METRICS, IMAGES_, Error_


    def Predict_Image(self, 
                      Image, 
                      show=False,
                      ):
        
        Models_ = self.Models
        cc_method = Models_.model_cc[2]  # location method used
        # print(cc_method)

        if isinstance(Image, str):
            Image_bgr = cv2.imread(Image)
            Image = to_float64(Image_bgr[:, :, ::-1])
        elif isinstance(Image, np.ndarray):
            Image = Image

        
        Models_ = self.Models

        time_start = time.time()

        # apply flat field correction
        if Models_.model_ffc is not None:
            ffc = FlatFieldCorrection()
            Image_bgr = to_uint8(Image[:, :, ::-1])
            Image_ffc = ffc.apply_ffc(img=Image_bgr, multiplier=Models_.model_ffc)
            Image_ffc = to_float64(Image_ffc[:, :, ::-1])
        else:
            Image_ffc = Image

        # apply gamma correction
        if Models_.model_gc is not None:
            Image_gc = predict_image(img=Image_ffc, coeffs=Models_.model_gc, ref_illuminant=self.REF_ILLUMINANT)
            Image_gc = np.clip(Image_gc, 0, 1)
        else:
            Image_gc = Image_ffc
    
        # apply white balance
        if Models_.model_wb is not None:
            Image_wb = Image_gc @ Models_.model_wb
            Image_wb = np.clip(Image_wb, 0, 1)
        else:
            Image_wb = Image_gc

        # apply color correction
        Image_cc = None
        if Models_.model_cc is not None:
            if cc_method == 'conv':
                Image_cc = colour.characterisation.apply_matrix_colour_correction(
                    RGB=Image_wb,
                    CCM=Models_.model_cc[0],
                    **Models_.model_cc[1]
                )
            elif cc_method == 'ours':
                Image_cc = predict_(RGB=Image_wb, M=Models_.model_cc[0])

            Image_cc = np.clip(Image_cc, 0, 1)


        time_end = time.time()

        log_(f"Time elapsed: {(time_end - time_start):.2f}s", 'light_green', 'bold', 'info')

        Images = {}
        Images['FFC'] = Image_ffc
        Images['GC'] = Image_gc
        Images['WB'] = Image_wb
        Images['CC'] = Image_cc

        if show:

            for k, v in Images.items():
                try:
                    colour.plotting.plot_image(v, title=k)
                except:
                    pass

        return Images

   