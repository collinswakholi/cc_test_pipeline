import os, re
import pandas as pd

from utils.logger_ import log_
from key_functions import plot_raincloud, get_stats, save_stats


FOLDER = 'Data/Light_Temperatures/Results/Plots'
comparisons = {
    'With': 'With_FFC_GC_WB_CC_Results',
    'Without': 'With_FFC_CC_Results',
}

plots_dir = os.path.join(FOLDER, 'Plots_Impact_GC_WB_Cal')
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

methods = ['linear', 'pls','nn', 'conv']
Degs = [1, 2]
Ligting = ['D65', 'D50']

SHOW = False
SAVE = True
DO_STATS = True

RUN_FOR_CAL = True # run for cal or val

# load with an without avrage data

Data = {
    'With': None,
    'Without': None,
}

for key, val in comparisons.items():
    Data[key] = pd.read_excel(os.path.join(FOLDER, val, 'Average_data.xlsx'))
    # log_(f'{key} data loaded - shape: {data[key].shape} /n/n {data[key].head()}', 'cyan', 'italic')

a=1


for light in Ligting:
    for deg_ in Degs:
        Plot_Data = pd.DataFrame()
        for im, method in enumerate(methods):
            log_(f'Processing method: {method}, Deg: {deg_}, Lighting: {light}...', 'magenta', 'italic')
            
            Data_ = Data.copy()
            for key, val in comparisons.items():
                Data_[key] = Data_[key][(Data_[key]['Method'] == method) & 
                                       (Data_[key]['Deg'] == deg_) & 
                                       (Data_[key]['Lighting'] == light)]

            if Data_['With'].empty or Data_['Without'].empty:
                log_(f"No data found for method: {method}, Deg: {deg_}, Lighting: {light}", 'red', 'italic', 'warn')
                continue

            data = Data_.copy()
            # keep columns with DeltaE and Tile
            data['With'] = data['With'][['Tile'] + [col for col in data['With'].columns if 'DeltaE' in col]]
            data['Without'] = data['Without'][['Tile'] + [col for col in data['Without'].columns if 'DeltaE' in col]]


            # make tile the index
            data['With'] = data['With'].set_index('Tile')
            data['Without'] = data['Without'].set_index('Tile')

            # keep only columns with names with 'val' and 'Original'
            if not RUN_FOR_CAL:
                data['With'] = data['With'][[col for col in data['With'].columns if 'val' in col]]
                data['Without'] = data['Without'][[col for col in data['Without'].columns if 'val' in col]]
            else:
                # keep only columns with names with 'Original' and 'CC' and no 'val'
                data['With'] = data['With'][[col for col in data['With'].columns if ('val' not in col) & ('CC' in col)]]
                data['Without'] = data['Without'][[col for col in data['Without'].columns if ('val' not in col) & ('CC' in col)]]

            data['With'].columns = [f'{im}_With_{col}' for col in data['With'].columns]
            data['With'] = data['With'].reset_index()

            data['Without'].columns = [f'{im}_Without_{col}' for col in data['Without'].columns]
            data['Without'] = data['Without'].reset_index()

            # merge with and without data on Tile and original data
            data_ = pd.merge(data['With'], data['Without'], how='outer', on=['Tile'])
            # rename columns by removing DeltaE_val
            if RUN_FOR_CAL:
                data_.columns = [re.sub(r"_CC_DeltaE", f"_{method}", col) for col in data_.columns]
            else:
                data_.columns = [re.sub(r"_CC_DeltaE_val", f"_{method}", col) for col in data_.columns]
            # log_(f'Data merged shape: {data_.shape} \n\n {data_.head()}', 'cyan', 'italic')


            if Plot_Data.empty:
                Plot_Data = data_
            else:
                Plot_Data = pd.merge(Plot_Data, data_, how='outer', on=['Tile'])

        
        # log_(f'Plot_data shape: {Plot_Data.shape} \n\n {Plot_Data.head()}', 'cyan', 'italic')

        tiles = Plot_Data['Tile'].tolist()
        Plot_Data = Plot_Data[[col for col in Plot_Data.columns if col != 'Tile']]

        title_name = f"DeltaE vs GC and WB (Deg: {deg_}, Lighting: {light})"

        Plot_Data = Plot_Data.sort_index(axis=1, ascending=True)

        # log_(f'Plot_data shape: {Plot_Data.shape} \n\n {Plot_Data.head()}', 'cyan', 'italic')

        plot_raincloud(
            Plot_Data,
            violin_=True, 
            condition_ = "col", 
            patch_names = tiles,
            y_label = "DeltaE", 
            x_label = "",
            title_name = title_name,
            palette_name="Paired",
            fig_size=(10, 8),
            show = SHOW,
            save_ = os.path.join(
                plots_dir,
                re.sub(r"[(),: ]", "_", title_name) + ".svg") if SAVE else None,
            )

            
        if DO_STATS:
            data__stat, anova_results, tukey_df, t_test_df = get_stats(data_, condition_="col")
            if SAVE:
                save_stats(data__stat, anova_results, tukey_df, t_test_df, plots_dir, title_name)

        a=1

            # save data
            # data_.to_excel(os.path.join(FOLDER, f'Comparison_{method}_{deg_}_{light}.xlsx'), index=False, float_format='%.15f')
