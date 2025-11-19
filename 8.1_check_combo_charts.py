# load images with 2 charts, check if they are the same.
# load images with or without FFC.

import cv2
import os, glob

from utils.logger_ import log_
from key_functions import extract_color_charts, get_illuminant_from, adapt_chart, to_float64, get_metrics, scatter_RGB

import colour
import colour.plotting
from utils.metrics_ import Metrics

import pandas as pd
import numpy as np



"""
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
VARIABLES (Only edit these)
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- 
"""

ILLUMINANT_FROM = 'D65'
CMFS = 'CIE 1931 2 Degree Standard Observer'

data_folder = 'Data/Light_Temperatures/Results'
add_str = '1Deg_nn'
light_temps = ['Amazon', 'Dalatin', 'D50', 'D65']

results_folder_ = os.path.join(data_folder, 'Combo_Test', f'ct_{add_str}')

image_folders = {}
for light_temp in light_temps:
    with_FFC = os.path.join(data_folder, f'With_FFC_GC_WB_CC_Results_{add_str}', light_temp)
    without_FFC = os.path.join(data_folder, f'With_CC_Results_{add_str}', light_temp)

    # if both folders exist
    if os.path.exists(with_FFC) and os.path.exists(without_FFC):
        image_folders[f'{light_temp}'] = {
            'With_FFC': with_FFC,
            'Without_FFC': without_FFC
        }

search_strs = ['Combo', '_CC.']

save_data = True
show = False


"""
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
GET REFERENCE CHART
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
"""

# Get refence colour chart from color
REF_ILLUMINANT = get_illuminant_from(ILLUMINANT_FROM, CMFS=CMFS)
REFERENCE_CHART = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
REFERENCE_CHART = adapt_chart(REFERENCE_CHART, REF_ILLUMINANT)
data = list(REFERENCE_CHART.data.values())
names = list(REFERENCE_CHART.data.keys())

REFERENCE_RGB = colour.XYZ_to_sRGB(
    colour.xyY_to_XYZ(data),    
    illuminant=REF_ILLUMINANT,
    apply_cctf_encoding=True
)
REFERENCE_RGB = np.clip(REFERENCE_RGB, 0, 1)


"""
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
MAIN LOOP
--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---
"""
DeltaEs_all = pd.DataFrame()
DeltaEs_average = pd.DataFrame()
Others_all = pd.DataFrame()
Charts_all = pd.DataFrame()
Charts_Average = pd.DataFrame() # average charts for each light temp

DeltaEs_all_2 = pd.DataFrame()
DeltaEs_average_2 = pd.DataFrame()

for Key_, Values_ in image_folders.items():
    results_folder__ = os.path.join(results_folder_, Key_)
    for key, values in Values_.items():

        log_(f'Working on {Key_} - {key} images from `{values}`', 'cyan', 'italic')
        results_folder = os.path.join(results_folder__, key)
        Image_list = glob.glob(os.path.join(values,  '*.jpg'), recursive=True)
        Image_list = [img for img in Image_list if search_strs[0] in img and search_strs[1] in img]

        log_(f'Images found: {len(Image_list)}', 'cyan', 'italic')

        if len(Image_list) == 0:
            log_(f'No images found for {key}', 'red', 'italic')
            continue

        for img in Image_list:
            img_no_FFC = cv2.imread(img)
            c_charts, img_draw = extract_color_charts(img_no_FFC, n_charts=2)

            log_(f'Charts found: {len(c_charts)}', 'cyan', 'italic')

            try:
                colour.plotting.plot_image(to_float64(img_draw[:,:,::-1]), title=f'{key} - {os.path.basename(img)}') if show else None
            except:
                pass

            if save_data:
                # print(results_folder)
                if not os.path.exists(results_folder):
                    os.makedirs(results_folder)

                cv2.imwrite(os.path.join(results_folder, os.path.basename(img)), img_draw)
            
            chart_for_plot = dict()
            Metrics_all = pd.DataFrame()
            Metrics_summary = pd.DataFrame()
            Metrics_other = pd.DataFrame()
            Charts = pd.DataFrame()

            for i, chart in enumerate(c_charts):
                chart_srgb = to_float64(chart)
                # compare with reference rgb
                metrics_ = get_metrics(REFERENCE_RGB, chart_srgb, illuminant=REF_ILLUMINANT)
                other_metrics = Metrics(
                    gt=REFERENCE_RGB,
                    pred=chart_srgb
                ).get_(return_df=True)

                log_(f'Metrics for Chart {i+1} vs Reference:\n{metrics_.Data_summary}', 'green', 'italic', 'info')

            
                metrics_all = metrics_.Data
                metrics_all.index = names
                metrics_all.columns = [f'{col}_Chart_{i+1}' for col in metrics_all.columns]
                # print(metrics_all)

                metrics_summary = metrics_.Data_summary
                metrics_summary.index = [f'Chart_{i+1}_vs_Reference']
                
                # print(metrics_summary)

                other_metrics_ = other_metrics.metrics
                other_metrics_.index = [f'Chart_{i+1}_vs_Reference']

                chart_df = pd.DataFrame(chart_srgb, columns=[f'R_{i+1}', f'G_{i+1}', f'B_{i+1}'], index=names)

                # update chart_for_plot dict
                chart_for_plot[f'Chart_{i+1}'] = chart_srgb

                #update metrics
                Metrics_all = pd.concat([Metrics_all, metrics_all], axis=1)
                Metrics_summary = pd.concat([Metrics_summary, metrics_summary], axis=0)
                Metrics_other = pd.concat([Metrics_other, other_metrics_], axis=0)
                Charts = pd.concat([Charts, chart_df], axis=1)


            save_str = os.path.join(results_folder, os.path.splitext(os.path.basename(img))[0]+f'_scatter_charts_vs_ref.svg') if save_data else None
            scatter_RGB(REFERENCE_RGB, chart_for_plot, save_=save_str) if show else None

            # 1. DEAL WITH CHARTS
            Charts.columns = [[f'Value_{col}' for col in Charts.columns]]
            # add column for ref_illuminant, light temperature, image name, and key to charts in positions 0, 1, 2, 3
            Charts.insert(0, 'Light', Key_)
            Charts.insert(1, 'Image', os.path.basename(img).split('.')[0])
            Charts.insert(2, 'FFC', key)

            Charts[[f'Ref_{c}' for c in ['R', 'G', 'B']]] = REFERENCE_RGB


            # put all charts in one dataframe
            Charts_all = pd.concat([Charts_all, Charts], axis=0)

            # 2. DEAL WITH METRICS (deltaE)
            # keep only deltaE metrics in Metrics_all
            Metrics_all = Metrics_all[[col for col in Metrics_all.columns if 'DeltaE_' in col]]
            Metrics_all.insert(0, 'Light', Key_)
            Metrics_all.insert(1, 'Image', os.path.basename(img).split('.')[0])
            Metrics_all.insert(2, 'FFC', key)

            Metrics_summary = Metrics_summary[[col for col in Metrics_summary.columns if 'DE_' in col]]
            Metrics_summary.insert(0, 'Light', Key_)
            Metrics_summary.insert(1, 'Image', os.path.basename(img).split('.')[0])
            Metrics_summary.insert(2, 'FFC', key)

            Metrics_other.insert(0, 'Light', Key_)
            Metrics_other.insert(1, 'Image', os.path.basename(img).split('.')[0])
            Metrics_other.insert(2, 'FFC', key)

            DeltaEs_all = pd.concat([DeltaEs_all, Metrics_all], axis=0)
            DeltaEs_average = pd.concat([DeltaEs_average, Metrics_summary], axis=0)
            Others_all = pd.concat([Others_all, Metrics_other], axis=0)
            
            if len(c_charts) == 2:
                chart1 = c_charts[0]
                chart2 = c_charts[1]

                chart1_srgb = to_float64(chart1)
                chart2_srgb = to_float64(chart2)

                metrics_ = get_metrics(chart1_srgb, chart2_srgb, illuminant=REF_ILLUMINANT)
                other_metrics = Metrics(
                    gt=chart1_srgb,
                    pred=chart2_srgb
                ).get_(return_df=True)

                log_(f'Metrics for Chart 1 vs Chart 2:\n{metrics_.Data_summary}', 'green', 'italic', 'info')

            
                metrics_all = metrics_.Data
                metrics_all.index = names
                metrics_summary = metrics_.Data_summary
                bn_ = os.path.splitext(os.path.basename(img))[0]

                deltaE_all_2 = metrics_all[[col for col in metrics_all.columns if 'DeltaE' in col]]
                deltaE_summary_2 = metrics_summary[[col for col in metrics_summary.columns if 'DE_' in col]]

                deltaE_all_2.insert(0, 'Light', Key_)
                deltaE_all_2.insert(1, 'Image', bn_)
                deltaE_all_2.insert(2, 'FFC', key)

                deltaE_summary_2.insert(0, 'Light', Key_)
                deltaE_summary_2.insert(1, 'Image', bn_)
                deltaE_summary_2.insert(2, 'FFC', key)

                DeltaEs_all_2 = pd.concat([DeltaEs_all_2, deltaE_all_2], axis=0)
                DeltaEs_average_2 = pd.concat([DeltaEs_average_2, deltaE_summary_2], axis=0)

                

                if show:
                    chart1_chart = colour.characterisation.ColourChecker(
                        name = "Chart 1",
                        data = dict(zip(names, colour.XYZ_to_xyY(colour.sRGB_to_XYZ(
                                chart1_srgb,
                                illuminant=REF_ILLUMINANT,
                            )))),
                        illuminant=REF_ILLUMINANT,
                        rows=4,
                        columns=6,
                    )

                    chart2_chart = colour.characterisation.ColourChecker(
                        name = "Chart 2",
                        data = dict(zip(names, colour.XYZ_to_xyY(colour.sRGB_to_XYZ(
                                chart2_srgb,
                                illuminant=REF_ILLUMINANT,
                            )))),
                        illuminant=REF_ILLUMINANT,
                        rows=4,
                        columns=6,
                    )

                    plot_ = colour.plotting.plot_multi_colour_checkers(
                        [chart1_chart, chart2_chart],
                        title=f'{key} - {os.path.basename(img)} - Charts comparison (Chart 1 vs Chart 2)',
                    )

                    fig_ = plot_[0]
                    fig_.savefig(os.path.join(results_folder, os.path.splitext(os.path.basename(img))[0]+f'_charts_comparison.svg'))

            else:
                log_(f'Charts found: {len(c_charts)}', 'light_yellow', 'italic')

            
    # get averages for lamps
if save_data:
    # save metrics in csv
    DeltaEs_all.to_csv(os.path.join(results_folder_, f'ALL_DeltaEs.csv'), float_format='%.9f', encoding='utf-8-sig')
    DeltaEs_average.to_csv(os.path.join(results_folder_, f'AVG_DeltaEs.csv'), float_format='%.9f', encoding='utf-8-sig')
    Others_all.to_csv(os.path.join(results_folder_, f'ALL_Others.csv'), float_format='%.9f', encoding='utf-8-sig')

    Charts_all.to_csv(os.path.join(results_folder_, f'ALL_Charts.csv'), float_format='%.9f', encoding='utf-8-sig')

    DeltaEs_all_2.to_csv(os.path.join(results_folder_, f'ALL_DeltaEs_2.csv'), float_format='%.9f', encoding='utf-8-sig')
    DeltaEs_average_2.to_csv(os.path.join(results_folder_, f'AVG_DeltaEs_2.csv'), float_format='%.9f', encoding='utf-8-sig')
