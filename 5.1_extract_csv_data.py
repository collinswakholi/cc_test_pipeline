import os, glob
import pandas as pd
from tqdm import tqdm
import re
from utils.logger_ import log_

data_path = 'Data/Light_Temperatures/Results'
search_str = 'With_FFC_GC_WB_CC_Results_' # repeat for 'With_FFC_GC_CC_Results_', 'With_FFC_WB_CC_Results_', 'With_CC_Results_', 'With_FFC_CC_Results_', 'With_FFC_GC_WB_CC_Results_'
save_str = search_str


# split the search string into parts, get the first operation on the image
first_op = search_str.split('_')[1]

def get_steps_deg_method(folder):
    remap = re.sub(r'With_|_Results_|Deg_', '#', folder)
    split_ = remap.split('#')
    return split_[1], split_[2], split_[3]


folders = [f for f in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, f))]
# remove any folders that don't contain the search string
folders_selected = [f for f in folders if search_str in f]
if len(folders_selected) == 0:
    log_(f'No folders found that contain {search_str}', 'red', 'italic', 'warn')
    exit()

log_(f'Found {len(folders_selected)} folders', 'cyan', 'italic')
[log_(f'{f}', 'magenta', 'italic') for f in folders_selected]


ALL_METRICS_PD = pd.DataFrame()
SUMMARY_METRICS_PD = pd.DataFrame()
ALL_METRICS_VAL_PD = pd.DataFrame()
SUMMARY_METRICS_VAL_PD = pd.DataFrame()
for folder in tqdm(folders_selected):

    folder_basename = os.path.basename(folder)

    steps, deg, method = get_steps_deg_method(folder)

    # get the .csv files in the folder and subfolders
    csv_files = glob.glob(os.path.join(data_path, folder, '**/*.csv'), recursive=True)

    All_metrics_csv_files = [f for f in csv_files if '_All_Metrics.csv' in f]
    Summary_metrics_csv_files = [f for f in csv_files if '_Summary_Metrics.csv' in f]

    All_metrics_csv_files_val = All_metrics_csv_files.copy()
    Summary_metrics_csv_files_val = Summary_metrics_csv_files.copy()

    All_metrics_csv_files = [f for f in All_metrics_csv_files if '_Corrected_' not in f]
    Summary_metrics_csv_files = [f for f in Summary_metrics_csv_files if '_Corrected_' not in f]
    # print(All_metrics_csv_files)

    # keep only the csv files that contain _Corrected_ in their name
    All_metrics_csv_files_val = [f for f in All_metrics_csv_files_val if '_Corrected_'  in f]
    Summary_metrics_csv_files_val = [f for f in Summary_metrics_csv_files_val if '_Corrected_' in f]
    # print(All_metrics_csv_files_val)

    All_metrics_df = pd.DataFrame()
    for csv_file in All_metrics_csv_files:
        file_base_folder = os.path.basename(os.path.dirname(csv_file))
        file_basename = os.path.basename(csv_file)
        file_basename = file_basename.replace('_All_Metrics.csv', '')
        df = None
        df = pd.read_csv(csv_file)
        if df.empty:
            log_(f'"{csv_file}" is empty', 'red', 'italic')
            continue
        try:
            df = df.rename(columns={df.columns[0]: 'Tile'})
        except:
            pass
        # remove columns whose column names contain 'Before_' except for those that contain 'Before_FFC'
        df = df.loc[:, ~df.columns.str.contains('Before_') | df.columns.str.contains(f'Before_{first_op}')]
        # print(df.columns)

        # check for column names that contain "_M1_" and "_M2_", rename both of them to "_"
        M1_ = False
        M2_ = False
        for col in df.columns:
            if '_M1_' in col:
                df.rename(columns={col: col.replace('_M1_', '_')}, inplace=True)
                M1_ = True

            if '_M2_' in col:
                df.rename(columns={col: col.replace('_M2_', '_')}, inplace=True)
                M2_ = True
        
        
        df.insert(0, 'Folder', folder_basename)
        df.insert(1, 'Lighting', file_base_folder)
        df.insert(2, 'Image', file_basename)
        df.insert(3, 'Step', steps)
        df.insert(4, 'Deg', int(deg))
        df.insert(5, 'Method', method)

        # print(df)
        if M1_ == True or M2_ == True:
            All_metrics_df = pd.concat([All_metrics_df, df], axis=0)
        
    
    # print(All_metrics_df.head())

    Summary_metrics_df = pd.DataFrame()
    for csv_file in Summary_metrics_csv_files:
        file_base_folder = os.path.basename(os.path.dirname(csv_file))
        file_basename = os.path.basename(csv_file)
        file_basename = file_basename.replace('_Summary_Metrics.csv', '')
        df = None
        df = pd.read_csv(csv_file)
        if df.empty:
            log_(f'"{csv_file}" is empty', 'red', 'italic')
            continue
        try: 
            df = df.rename(columns={df.columns[0]: 'Tile'})
        except:
            pass
        # remove column 1
        df.drop(df.columns[1], axis=1, inplace=True)
        # remove rows that contain 'Before_' except for those that contain 'Before_FFC' in row 1
        # print(df.head())
        df = df.loc[~df.iloc[:, 0].str.contains('Before_') | df.iloc[:, 0].str.contains('Before_FFC')]
        M_=False
        if df.iloc[:, 0].str.contains('_M1').any() or df.iloc[:, 0].str.contains('_M2').any():
            M_=True

        # print(df)
        df.insert(0, 'Folder', folder_basename)
        df.insert(1, 'Lighting', file_base_folder)
        df.insert(2, 'Image', file_basename)
        df.insert(3, 'Step', steps)
        df.insert(4, 'Deg', int(deg))
        df.insert(5, 'Method', method)
        
        if M_ == True:
            Summary_metrics_df = pd.concat([Summary_metrics_df, df], axis=0)
    
    # print(Summary_metrics_df.head())

    All_metrics_df_val = pd.DataFrame()
    for csv_file in All_metrics_csv_files_val:
        file_base_folder = os.path.basename(os.path.dirname(csv_file))
        file_basename = os.path.basename(csv_file)
        file_basename = file_basename.replace('_Corrected_All_Metrics.csv', '')
        df = None
        df = pd.read_csv(csv_file)
        if df.empty:
            log_(f'Empty file: {csv_file}', 'red', 'italic')
            continue

        # rename column 0 to 'Tile'
        tiles = df.columns[0]
        try:
            df.rename(columns={tiles: 'Tile'}, inplace=True)
        except:
            pass
        # df = df.rename(columns={df.columns[0]: 'Tile'})

        df.insert(0, 'Folder', folder_basename)
        df.insert(1, 'Lighting', file_base_folder)
        df.insert(2, 'Image', file_basename)
        df.insert(3, 'Step', steps)
        df.insert(4, 'Deg', int(deg))
        df.insert(5, 'Method', method)

        All_metrics_df_val = pd.concat([All_metrics_df_val, df], axis=0)

    #     # print(df)
        
    
    Summary_metrics_df_val = pd.DataFrame()
    for csv_file in Summary_metrics_csv_files_val:
        # if 'Combo' not in csv_file:
        #     continue
        file_base_folder = os.path.basename(os.path.dirname(csv_file))
        file_basename = os.path.basename(csv_file)
        file_basename = file_basename.replace('_Corrected_Summary_Metrics.csv', '')
        df = None
        df = pd.read_csv(csv_file)
        if df.empty:
            log_(f'Empty file: {csv_file}', 'red', 'italic')
            continue

        df = df.drop(df.columns[0], axis=1)
        df.insert(0, 'Folder', folder_basename)
        df.insert(1, 'Lighting', file_base_folder)
        df.insert(2, 'Steps', steps)
        df.insert(3, 'Image', file_basename)
        df.insert(4, 'Deg', int(deg))
        df.insert(5, 'Method', method)
        # print(df)
        Summary_metrics_df_val = pd.concat([Summary_metrics_df_val, df], axis=0)

    # print(Summary_metrics_df_val)

    

    ALL_METRICS_PD = pd.concat([ALL_METRICS_PD, All_metrics_df], axis=0)
    SUMMARY_METRICS_PD = pd.concat([SUMMARY_METRICS_PD, Summary_metrics_df], axis=0)

    ALL_METRICS_VAL_PD = pd.concat([ALL_METRICS_VAL_PD, All_metrics_df_val], axis=0)
    SUMMARY_METRICS_VAL_PD = pd.concat([SUMMARY_METRICS_VAL_PD, Summary_metrics_df_val], axis=0)


log_(f"Saving all metrics to '{os.path.join(data_path, save_str +'All_Metrics.xlsx')}'", 'green', 'bold')
ALL_METRICS_PD.to_excel(os.path.join(data_path, save_str + 'All_Metrics.xlsx'), index=False, float_format='%.12f')

log_(f"Saving summary metrics to '{os.path.join(data_path, save_str + 'Summary_Metrics.xlsx')}'", 'green', 'bold')
SUMMARY_METRICS_PD.to_excel(os.path.join(data_path, save_str + 'Summary_Metrics.xlsx'), index=False, float_format='%.12f')

log_(f"Saving all metrics to '{os.path.join(data_path, save_str +'All_Metrics_val.xlsx')}'", 'green', 'bold')
ALL_METRICS_VAL_PD.to_excel(os.path.join(data_path, save_str + 'All_Metrics_val.xlsx'), index=False, float_format='%.12f')  

log_(f"Saving summary metrics to '{os.path.join(data_path, save_str + 'Summary_Metrics_val.xlsx')}'", 'green', 'bold') 
SUMMARY_METRICS_VAL_PD.to_excel(os.path.join(data_path, save_str + 'Summary_Metrics_val.xlsx'), index=False, float_format='%.12f')
    
log_(f'Done...', 'green', 'Bold', 'info')