import pandas as pd
import numpy as np
import os
import re

from key_functions import plot_raincloud, get_stats, save_stats, other_plots
from utils.logger_ import log_


"""
#######################################################################################################################
VARIABLES (Only edit these)
#######################################################################################################################
"""

SHOW = False
SAVE = True
DO_STATS = True
PLOT_VAL = True
PLOT_CAL = True

file_Str = 'With_FFC_GC_WB_CC_Results_All_Metrics'

data_dir = "Data/Light_Temperatures/Results"
data_file = f"{file_Str}.xlsx"
val_file = f"{file_Str}_val.xlsx"
plots_dir = os.path.join(data_dir, 'Plots', data_file.split('_All_Metrics')[0])


# Don't Change this
first_op = data_file.split('_')[1]

"""
#######################################################################################################################
Data Preprocessing
#######################################################################################################################
"""

def read_excel_fast(file_path):
    # Create a corresponding pickle file path
    pickle_path = file_path.replace('.xlsx', '.pkl')
    
    if os.path.exists(pickle_path):
        # If the pickle exists, load from it
        df = pd.read_pickle(pickle_path)
    else:
        # Otherwise, read the Excel file and then save a pickle version
        df = pd.read_excel(file_path, engine='openpyxl')
        df.to_pickle(pickle_path)
    return df.dropna()

if not os.path.exists(os.path.join(plots_dir,'Stats')):
    os.makedirs(os.path.join(plots_dir,'Stats'))

data = read_excel_fast(os.path.join(data_dir, data_file))
data_val = read_excel_fast(os.path.join(data_dir, val_file))
data_test = data_val[~data_val['Image'].isin(data['Image'])]
data_val = data_val[data_val['Image'].isin(data['Image'])]


lighting = data['Lighting'].values
unique_lighting = np.unique(lighting)

image_names = data['Image'].values
image_names_val = data_val['Image'].values
image_names_test = data_test['Image'].values
unique_image_names = np.unique(image_names)
unique_image_names_val = np.unique(image_names_val)
unique_image_names_test = np.unique(image_names_test)

image_folders = data['Folder'].values
unique_image_folders = np.unique(image_folders)

methods = data['Method'].values
Degs = data['Deg'].values
unique_Degs = np.unique(Degs)
unique_methods = np.unique(methods)

log_(f'Unique Lightings: {unique_lighting}', 'magenta', 'italic')
log_(f'Unique Methods: {unique_methods}', 'magenta', 'italic')
log_(f'Unique Degs: {unique_Degs}', 'magenta', 'italic')
log_(f'Number of unique images: {len(unique_image_names)}', 'magenta', 'italic')
log_(f'Number of unique images_val: {len(unique_image_names_val)}', 'magenta', 'italic')
log_(f'Number of unique images_test: {len(unique_image_names_test)}', 'magenta', 'italic')
log_(f'Number of unique image folders: {len(unique_image_folders)}', 'magenta', 'italic')


all_patch_names = data['Tile'].values
patch_names = np.unique(all_patch_names)

# rename the columns that contain name_strings_before to steps + other remaining string
name_stings_before_after = [f'Before_{first_op}', 'After_FFC', 'After_GC', 'After_WB', 'After_CC']
steps = ['Original', 'FFC', 'GC', 'WB', 'CC']

for name_string, step in zip(name_stings_before_after, steps):
    data = data.rename(columns={col: f"{step}{col.split(name_string)[1]}" for col in data.columns if name_string in col})

data_val = data_val.rename(columns={'DeltaE': 'CC_DeltaE_val', 'MSE': 'CC_MSE_val', 'MAE': 'CC_MAE_val'})
ids_1 = data_val['Chart'].values == 1
data_val = data_val[ids_1].drop(['Chart'], axis=1)

common_keys = ['Folder', 'Lighting', 'Image', 'Tile', 'Step', 'Method', 'Deg']
Appended_data = pd.merge(data, data_val, how='outer', on=common_keys)

Appended_data.to_excel(os.path.join(plots_dir, 'Appended_data.xlsx'), index=False, na_rep='NaN')

# drop rows with NaN values
Appended_data = Appended_data.dropna()
data = Appended_data

def get_average_data(Appended_data):
    Average_data = pd.DataFrame()
    data = Appended_data
    
    for folder in data['Folder'].unique():
        folder_data = data[data['Folder'] == folder]
        
        for lighting in folder_data['Lighting'].unique():
            lighting_data = folder_data[folder_data['Lighting'] == lighting]
            
            all_image_data = []
            for img in lighting_data['Image'].unique():
                img_data = lighting_data[lighting_data['Image'] == img]
                
                deg = img_data['Deg'].values[0]
                method = img_data['Method'].values[0]
                
                img_data = img_data[['Tile'] + [col for col in img_data.columns if 'DeltaE' in col or 'MSE' in col or 'MAE' in col]]
                img_data = img_data.groupby('Tile', as_index=False).mean()
                
                all_image_data.append(img_data)
            
            all_image_data_df = pd.concat(all_image_data, ignore_index=True).groupby('Tile', as_index=False).mean()
            all_image_data_df.insert(1, 'Folder', folder)
            all_image_data_df.insert(2, 'Lighting', lighting)
            all_image_data_df.insert(3, 'Number_of_images', len(lighting_data['Image'].unique()))
            all_image_data_df.insert(4, 'Deg', deg)
            all_image_data_df.insert(5, 'Method', method)

            
            Average_data = pd.concat([Average_data, all_image_data_df], ignore_index=True)
    
    return Average_data

Average_data = get_average_data(Appended_data)

log_(f'Average_data - shape:{list(Average_data.shape)} \n\n{Average_data.head()}', 'cyan', 'italic')

Average_data.to_excel(os.path.join(plots_dir, 'Average_data.xlsx'), 
                      index=False, float_format='%.15f')

"""
#######################################################################################################################
SECTION 1. Plots and Stats for lamps (Amazon, Dalatlin, D50, D65)
#######################################################################################################################
"""
PLOT_CAL = True
PLOT_VAL = True
strs_ = ['DeltaE', 'DeltaE_val']
multi_ = [s for s, flag in zip(strs_, [PLOT_CAL, PLOT_VAL]) if flag]

# Section 1. Plots for lamps (Amazon, Dalatlin, D50, D65) # use Average_data
for deg in unique_Degs:
    for method in unique_methods:

        deg_method_data = Average_data[(Average_data['Deg'] == deg) & (Average_data['Method'] == method)]
        if deg_method_data.shape[0] == 0:
            continue
        
        selected_columns = [col for col in deg_method_data.columns if any(s in col for s in multi_) or ('Original_DeltaE' in col)] if multi_ else []

        if not PLOT_VAL:
            selected_columns = [col for col in selected_columns if not any(s in col for s in strs_[1:])]
        

        selected_columns = list(set(selected_columns))
        if not selected_columns or len(selected_columns) < 1:
            continue

        selected_cols = ['Lighting', 'Tile'] + selected_columns

        DeltaE_lighting = deg_method_data[selected_cols].set_index('Lighting')

        after_deltae_cols = [col for col in DeltaE_lighting.columns if 'CC_' in col]

        if len(after_deltae_cols)>1:
            after_deltae_cols = [col for col in after_deltae_cols if 'val' in col]
            if len(after_deltae_cols)>1:
                log_(f"Unexpected number of DeltaE columns for deg {deg} and method {method}", 'red')
                continue

        before_deltae_cols = [col for col in DeltaE_lighting.columns if 'Original' in col]


        After_CC_DeltaE = DeltaE_lighting[after_deltae_cols + ['Tile']]
        Before_CC_DeltaE = DeltaE_lighting[before_deltae_cols + ['Tile']]
        
        # Append suffixes to index labels for clarity
        Before_CC_DeltaE.index = [str(idx) + "_Before" for idx in Before_CC_DeltaE.index]
        After_CC_DeltaE.index = [str(idx) + "__After" for idx in After_CC_DeltaE.index]  
        
        # Rename columns to a consistent format
        After_CC_DeltaE.columns = ["DeltaE_values", "Tile"]
        Before_CC_DeltaE.columns = ["DeltaE_values", "Tile"]

        concact_before_after = pd.concat([Before_CC_DeltaE, After_CC_DeltaE], axis=0)
        
        # sort data according to lighting (index),
        concact_before_after = concact_before_after.sort_index()

        tiles = concact_before_after['Tile'].tolist()
        concact_before_after = concact_before_after[[col for col in concact_before_after.columns if col != 'Tile']]

        # plot 1. Box-scatter, DeltaE vs Lighting (Before and after CC, for each lighting) ---- completed
        tile_name = f"1 DeltaE vs Lighting (Deg: {deg}, Method: {method})"
        
        plot_raincloud(
            concact_before_after, 
            violin_=True, 
            condition_ = "row", 
            patch_names = tiles,
            palette_name = "Paired" if (PLOT_CAL or PLOT_VAL) else None,
            y_label = "Mean DeltaE", 
            x_label = "Lighting",
            title_name = tile_name,
            show = SHOW,
            save_ = os.path.join(
                plots_dir, 
                re.sub(r"[(),: ]", "_", tile_name) + ".svg") if SAVE else None,
            )
        a=1
        if DO_STATS:
            data__stat, anova_results, tukey_df, t_test_df = get_stats(concact_before_after, condition_="row")
            if SAVE:
                save_stats(data__stat, anova_results, tukey_df, t_test_df, plots_dir, tile_name)

        # plot 2. Box-scatter-violins, Conditions/steps vs DeltaE (Before CC, after every step, for each lighting)
        Step_data = pd.DataFrame()
        anova_results = pd.DataFrame()
        tukey_df = pd.DataFrame()
        data_4_stats = pd.DataFrame()
        t_test_df = pd.DataFrame()

        
        for lighting_ in unique_lighting:
            L_DeltaE_ = DeltaE_lighting[DeltaE_lighting.index == lighting_]

            # log_(f"L_DeltaE_ - shape:{list(L_DeltaE_.shape)} \n\n{L_DeltaE_.head()}", 'cyan', 'italic')
            L_DeltaE_ = L_DeltaE_.reset_index(drop=True)
            L_DeltaE_.columns = [f"{lighting_}_{col}" if col != 'Tile' else col for col in L_DeltaE_.columns]
            L_DeltaE_.columns = [re.sub(r"_DeltaE", "", col) for col in L_DeltaE_.columns]  

            Step_data = L_DeltaE_ if Step_data.empty else pd.merge(Step_data, L_DeltaE_, how='outer', on=['Tile'])


            if DO_STATS:
                L_DeltaE_ = L_DeltaE_[[col for col in L_DeltaE_.columns if col != 'Tile']]
                data_, anova_results_, tukey_df_, t_test_df_ = get_stats(L_DeltaE_, condition_="col")
                add_str = f"{lighting_}_"
                anova_results_.index = [index + add_str for index in anova_results_.index]
                t_test_df_.index = [str(index) + add_str for index in t_test_df_.index]
                anova_results = pd.concat([anova_results, anova_results_], axis=0)
                tukey_df = pd.concat([tukey_df, tukey_df_], axis=0)
                data_4_stats = pd.concat([data_4_stats, data_], axis=0)
                t_test_df = pd.concat([t_test_df, t_test_df_], axis=0)


        tiles = Step_data['Tile'].tolist()
        Step_data = Step_data[[col for col in Step_data.columns if col != 'Tile']]
        Step_data.columns = [re.sub(r'_Original', "_1_Original", col) for col in Step_data.columns]
        Step_data.columns = [re.sub(r'_FFC', "_2_FFC", col) for col in Step_data.columns]
        Step_data.columns = [re.sub(r'_GC', "_3_GC", col) for col in Step_data.columns]
        Step_data.columns = [re.sub(r'_WB', "_4_WB", col) for col in Step_data.columns]
        Step_data.columns = [re.sub(r'_CC_val', "_6_Val", col) for col in Step_data.columns]
        Step_data.columns = [re.sub(r'_CC', "_5_CC", col) for col in Step_data.columns]

        # Step_data.index = patch_names
        title_name = f"2 DeltaE vs Steps (Deg: {deg}, Method: {method})"

        # rearrange the columns in alphabetical order
        Step_data = Step_data.sort_index(axis=1, ascending=True)

        plot_raincloud(
            Step_data,
            violin_=True, 
            condition_ = "col", 
            patch_names = tiles,
            y_label = "DeltaE", 
            x_label = "Steps",
            title_name = title_name,
            fig_size=(20, 10),
            show = SHOW,
            save_ = os.path.join(
                plots_dir,
                re.sub(r"[(),: ]", "_", title_name) + ".svg") if SAVE else None,
            )
        a=1
        if SAVE and DO_STATS:
            save_stats(data_4_stats, anova_results, tukey_df, t_test_df, plots_dir, title_name)

"""
#######################################################################################################################
SECTION 2. Plots/ Stats for images of different chart sizes (with small, big, or Combo charts)
#######################################################################################################################
"""

# Section 2. Plots for images of different chart sizes (with small, big, or Combo charts) 
# # use data from D50 (we got 3 small, 3 big, 3 Combo)
light_ = 'D65'
D50_data = data[data['Lighting'] == light_]
chart_sizes = ['small', 'big',]

PLOT_CAL = True
PLOT_VAL = False
strs_ = ['DeltaE', 'DeltaE_val']
multi_ = [s for s, flag in zip(strs_, [PLOT_CAL, PLOT_VAL]) if flag]

for deg in unique_Degs:
    for method in unique_methods:
        method_deg_data = D50_data[(D50_data['Deg'] == deg) & (D50_data['Method'] == method)]
        if method_deg_data.empty:
                continue
        
        Chart_df = pd.DataFrame()
        anova_results = pd.DataFrame()
        tukey_df = pd.DataFrame()
        data_4_stats = pd.DataFrame()
        t_test_df = pd.DataFrame()
        for chart_size in chart_sizes:
            chart_data = method_deg_data[method_deg_data['Image'].str.contains(chart_size, case=False)]
    
            if chart_data.empty:
                continue
            
            DeltaE_data = pd.DataFrame()
            DeltaE_clean = pd.DataFrame()
            DeltaE_clean_mean = pd.DataFrame()

            selected_columns = [col for col in chart_data.columns if any(s in col for s in multi_) or ('Original_DeltaE' in col)] if multi_ else []

            if not PLOT_VAL:
                selected_columns = [col for col in selected_columns if not any(s in col for s in strs_[1:])]
            

            selected_columns = list(set(selected_columns))
            if not selected_columns or len(selected_columns) < 1:
                continue

            # remove all columns except those with DeltaE and Before, After_CC in their names, File, and Folder
            DeltaE_data = chart_data[['Tile'] + selected_columns]
            # for column name with DeltaE,  remove all except those that contain 'Before_' or 'After_CC'
            DeltaE_clean = DeltaE_data[['Tile'] + [col for col in DeltaE_data.columns if 'Original_' in col or 'CC_' in col]]

            # for each different file name, get the mean DeltaE values for each patch
            DeltaE_clean = DeltaE_clean.set_index('Tile')
            DeltaE_clean_mean = DeltaE_clean.groupby('Tile').mean()

            add_str = f"{chart_size.upper()}_"

            # rename columns to include the chart size, degree, and method
            DeltaE_clean_mean.columns = [add_str + col for col in DeltaE_clean_mean.columns]

            Chart_df = pd.concat([Chart_df, DeltaE_clean_mean], axis=1)

            if DO_STATS:
                # reset indeces from patch names to numbers
                DeltaE_clean_mean = DeltaE_clean_mean.reset_index()
                data_, anova_results_, tukey_df_, t_test_df_ = get_stats(DeltaE_clean_mean, condition_="col")
                anova_results_.index = [index + add_str for index in anova_results_.index]
                t_test_df_.index = [str(index) + add_str for index in t_test_df_.index]
                anova_results = pd.concat([anova_results, anova_results_], axis=0)
                tukey_df = pd.concat([tukey_df, tukey_df_], axis=0)
                data_4_stats = pd.concat([data_4_stats, data_], axis=0)
                t_test_df = pd.concat([t_test_df, t_test_df_], axis=0)

    
        title_name = f"3 DeltaE vs Chart Size (Lighting: {light_}, Deg: {deg}, Method: {method})"
        # Rearrange the row indeces to match the patch names
        Chart_df = Chart_df.sort_index(axis=1, ascending=False)
        Chart_df = Chart_df.reindex()
        tile_names = list(Chart_df.index)
        palette_name="Paired"
        if (PLOT_CAL==True and PLOT_VAL==True):
            palette_name=None
        plot_raincloud(
            Chart_df, 
            violin_=True, 
            condition_="col",
            patch_names= tile_names,
            y_label="DeltaE", 
            x_label="Chart",
            palette_name=palette_name,
            title_name=title_name,
            show=SHOW,
            save_=os.path.join(
                plots_dir,
                re.sub(r"[(),: ]", "_", title_name) + ".svg") if SAVE else None,
        )
        
        if SAVE and DO_STATS:
            save_stats(data_4_stats, anova_results, tukey_df, t_test_df, plots_dir, title_name)

# a=1
"""
#######################################################################################################################
SECTION 3. Plots for steps (FFC, GC, WB, CC) # use Average_data from D65 and D50
#######################################################################################################################
"""

# Section 3. Plots for steps (FFC, GC, WB, CC) # use Average_data from D65 and D50
selected_lights = ['D65', 'D50']


anova_results = pd.DataFrame()
tukey_df = pd.DataFrame()
data_4_stats = pd.DataFrame()
t_test_df = pd.DataFrame()

for deg in unique_Degs:
    for method in unique_methods:
        for light in selected_lights:
            step_data = pd.DataFrame()
            step_data = Average_data[(Average_data['Lighting'] == light) & 
                                     (Average_data['Deg'] == deg) & 
                                     (Average_data['Method'] == method)]
            if step_data.empty:
                log_(f"No data found for Lighting: {light}, Deg: {deg}, Method: {method}", 'red', 'italic', 'warn')
                continue
            
            selected_columns = [col for col in step_data.columns if 'DeltaE' in col]
            Step_DeltaE = pd.DataFrame()
            Step_DeltaE_mean = pd.DataFrame()

            Step_DeltaE = step_data[['Tile'] + selected_columns].set_index('Tile')
            Step_DeltaE_mean = Step_DeltaE.groupby('Tile').mean()
            
            title_name = f"4 DeltaE vs Steps2 (Lighting: {light}, Deg: {deg}, Method: {method})"
            patch_names = list(Step_DeltaE_mean.index)
            plot_raincloud(
                Step_DeltaE_mean, 
                violin_=True, 
                condition_="col",
                patch_names= patch_names, 
                palette_name="Paired" if (PLOT_CAL==True and PLOT_VAL==True) else None,
                y_label="Mean DeltaE", 
                x_label="Steps",
                title_name=title_name,
                fig_size=(15,12),
                show=SHOW,
                save_=os.path.join(
                    plots_dir,
                    re.sub(r"[(),: ]", "_", title_name) + ".svg") if SAVE else None,
            )
            
            if DO_STATS:
                Step_DeltaE_mean = Step_DeltaE_mean.reset_index()
                data_, anova_results_, tukey_df_, t_test_df_ = get_stats(Step_DeltaE_mean, condition_="col")
                anova_results_.index = [index + f"(Lighting: {light}, Deg: {deg}, Method: {method})" for index in anova_results_.index]
                t_test_df_.index = [str(index) + f"(Lighting: {light}, Deg: {deg}, Method: {method})" for index in t_test_df_.index]
                anova_results = pd.concat([anova_results, anova_results_], axis=0)
                tukey_df = pd.concat([tukey_df, tukey_df_], axis=0)
                data_4_stats = pd.concat([data_4_stats, data_], axis=0)
                t_test_df = pd.concat([t_test_df, t_test_df_], axis=0)

name_ = "4_Stats for Each Step"
if SAVE and DO_STATS:
    save_stats(data_4_stats, anova_results, tukey_df, t_test_df, plots_dir, name_)

"""
#######################################################################################################################
SECTION 4. Plots for method (conv, linear, pls, nn), and complexity (degree) # Use Average_data from D65 and D50
#######################################################################################################################
"""

# # Section 4. Plots for method (conv, linear, pls, nn), and complexity (degree) # Use Average_data from D65 and D50
methods_ = unique_methods
selected_lights = ['D65', 'D50']

PLOT_CAL = True
PLOT_VAL = True

strs_ = ['CC_DeltaE', 'CC_DeltaE_val']
multi_ = [s for s, flag in zip(strs_, [PLOT_CAL, PLOT_VAL]) if flag]

for light in selected_lights:
    for deg in unique_Degs: 

        Method_df = pd.DataFrame()

        for i, method in enumerate(methods_):
            method_data = pd.DataFrame()
            method_data = Average_data[(Average_data['Lighting'] == light) & 
                                       (Average_data['Deg'] == deg) & 
                                       (Average_data['Method'] == method)]
            if method_data.empty:
                continue

            selected_columns = [col for col in method_data.columns if any(s in col for s in multi_)] if multi_ else []
            if not selected_columns or len(selected_columns) < 1:
                continue


            Method_DeltaE = pd.DataFrame()
            Method_DeltaE = method_data[['Tile'] + selected_columns].set_index('Tile')

            Method_DeltaE.columns = [f'{i+1}. {method}_' + col for col in Method_DeltaE.columns]
            Method_df = pd.concat([Method_df, Method_DeltaE], axis=1)

        title_name = f"5 DeltaE vs Fit Methods (Lighting: {light}, Deg: {deg})"
        Method_df.reindex()
        patch_names = list(Method_df.index)
        # print(Method_df.head())
        palette_name = "Paired"
        if PLOT_CAL==True and PLOT_VAL==True:
            palette_name = None
        plot_raincloud(
            Method_df, 
            violin_=True, 
            condition_="col", 
            patch_names= patch_names,
            palette_name=palette_name,
            y_label="Mean DeltaE", 
            x_label="Method",
            title_name=title_name,
            fig_size=(20,15),
            show=SHOW,
            save_=os.path.join(
                plots_dir,
                re.sub(r"[(),: ]", "_", title_name) + ".svg") if SAVE else None,
        )

        if DO_STATS:
            Method_df = Method_df.reset_index()
            data_, anova_results_, tukey_df_, t_test_df_ = get_stats(Method_df, condition_="col")
            anova_results_.index = [index + f"(Lighting: {light}, Deg: {deg})" for index in anova_results_.index]
            t_test_df_.index = [str(index) + f"(Lighting: {light}, Deg: {deg})" for index in t_test_df_.index]

        if SAVE and DO_STATS:
            save_stats(data_, anova_results_, tukey_df_, t_test_df_, plots_dir, title_name)

# a=1

"""
#######################################################################################################################
SECTION 5. Plots for degree of complexity (2, 3, 4,5 ) # Use Average_data from D65 and D50
#######################################################################################################################
"""

# section 5 (6). Plots for degree of complexity (2, 3, 4, 5) # Use Average_data from D65 and D50
degrees_ = unique_Degs
selected_lights = ['D65', 'D50']
methods_ = ['linear', 'nn', 'conv', 'pls']

PLOT_CAL = True
PLOT_VAL = True
strs_ = ['CC_DeltaE', 'CC_DeltaE_val']
multi_ = [s for s, flag in zip(strs_, [PLOT_CAL, PLOT_VAL]) if flag]

for light in selected_lights:
    for method in methods_:

        Degrees_df = pd.DataFrame()

        for i, deg in enumerate(degrees_):
            deg_data = pd.DataFrame()
            deg_data = Average_data[(Average_data['Lighting'] == light) & 
                                    (Average_data['Deg'] == deg) & 
                                    (Average_data['Method'] == method)]
            if deg_data.empty:
                continue

            selected_columns = [col for col in deg_data.columns if any(s in col for s in multi_)] if multi_ else []
            if not selected_columns or len(selected_columns) < 1:
                continue

            Deg_DeltaE = deg_data[['Tile'] + selected_columns].set_index('Tile')

            Deg_DeltaE.columns = [f'{i+1}. {deg}_Degree_' + col for col in Deg_DeltaE.columns]
            Degrees_df = pd.concat([Degrees_df, Deg_DeltaE], axis=1)

        title_name = f"6 DeltaE vs Degree of Complexity (Lighting: {light}, Method: {method})"
        patch_names = list(Degrees_df.index)
        palette_name = "Paired"
        if PLOT_CAL==True and PLOT_VAL==True:
            palette_name = None
        plot_raincloud(
            Degrees_df, 
            violin_=True, 
            condition_="col", 
            patch_names= patch_names,
            palette_name=palette_name,
            y_label="Mean DeltaE", 
            x_label="Degrees",
            title_name=title_name,
            fig_size=(20, 15),
            show=SHOW,
            save_=os.path.join(
                plots_dir,
                re.sub(r"[(),: ]", "_", title_name) + ".svg") if SAVE else None,
        ) 

        if DO_STATS:
            Degrees_df = Degrees_df.reset_index()
            data_, anova_results_, tukey_df_, t_test_df_ = get_stats(Degrees_df, condition_="col")
            anova_results_.index = [index + f"(Lighting: {light}, Method: {method})" for index in anova_results_.index]
            t_test_df_.index = [str(index) + f"(Lighting: {light}, Method: {method})" for index in t_test_df_.index]

        if SAVE and DO_STATS:
            save_stats(data_, anova_results_, tukey_df_, t_test_df_, plots_dir, title_name)
"""
#######################################################################################################################
SECTION 6. Other Plots/Stats
#######################################################################################################################
"""
title_name = "7 Other Plots" if SAVE else None
other_plots(Average_data, save_path=plots_dir, title_name=title_name, SHOW_=SHOW)

log_("Done...", 'light_green', 'Bold')