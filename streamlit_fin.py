import streamlit as st
import pandas as pd
import xgboost as xgb
import pickle
import numpy as np
import shap
import base64
from io import BytesIO
import matplotlib.pyplot as plt


# Load the model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the data
df = pd.read_csv('MstRecentElo_2023.csv')
df['fid'] = df['name'].astype(str) + "-" + df['fighter_id'].astype(str)

# Streamlit app
st.title('MMA Fighter Prediction Lookback 2023')

fighters = df['fid'].unique()

# Dropdowns for fighter selection
fighter1 = st.selectbox('Select Fighter 1', fighters, index=0)
fighter2 = st.selectbox('Select Fighter 2', fighters, index=1)

if st.button('Predict'):
    # Filter the data for the selected fighters
    mask1 = (df['fid'] == fighter1) | (df['fid'] == fighter2)
    df_filtered = df[mask1]
    df_filtered['fight_pk'] = 1
    df_filtered['elo'] = df_filtered['Current_Elo']
    
    df_filtered['AgeDiff'] = df_filtered.groupby('fight_pk')['Age'].diff()
    df_filtered['AgeDiff'] = df_filtered['AgeDiff'].fillna(df_filtered['AgeDiff'].max() * -1)
    
    df_filtered['EloDiff'] = df_filtered.groupby('fight_pk')['elo'].diff()
    df_filtered['EloDiff'] = df_filtered['EloDiff'].fillna(df_filtered['EloDiff'].max() * -1)
    
    df_diff = df_filtered
    
    def fill_and_adjust_column(df, column_name, group_by_column):
        """
        Fills missing values in a specified column by backward filling within each group defined by another column.
        It then adjusts the values of the specified column for the originally missing values to be negative.
        
        Parameters:
        - df: A pandas DataFrame.
        - column_name: The name of the column to fill and adjust.
        - group_by_column: The name of the column to group by for filling operations.
        
        Returns:
        - The DataFrame with the specified column modified as described.
        """
        mask = df[column_name].isna()
        
        df[f'{column_name}_diff'] = df.groupby(group_by_column)[column_name].transform(lambda x: x.bfill())
        
        df.loc[mask, f'{column_name}_diff'] = df.loc[mask, f'{column_name}_diff'] * -1
        
        return df

    def calculate_stat_diff(df, stat_col1, stat_col2, group_col, new_col_name):
        """
        For each fight, calculate the difference between two statistics which are in different columns and rows,
        and store the result in a new column in the same DataFrame.
        
        Parameters:
        - df: DataFrame containing the fight data.
        - stat_col1: The name of the first statistic column.
        - stat_col2: The name of the second statistic column.
        - group_col: The column by which to group the DataFrame (e.g., 'fight_id').
        - new_col_name: The name for the new column to be created with the calculated difference.
        
        Returns:
        - The same DataFrame with the new column added.
        """
        df.sort_values(by=[ 'date', group_col], inplace=True)
        
        fight_groups = df.groupby(group_col)
        if any(len(group) != 2 for _, group in fight_groups):
            raise ValueError("Each fight_id group must have exactly two rows.")
        
        df[new_col_name] = pd.Series(dtype=float)
        
        for _, group in fight_groups:
            if len(group) == 2:
                diff = group.iloc[0][stat_col1] - group.iloc[1][stat_col2]
                diff2 = group.iloc[1][stat_col1] - group.iloc[0][stat_col2]
                df.loc[group.index, new_col_name] = diff, diff2
        
        return df
    
    columns_to_include = [
    'pytime', 'RingTime', 'TotalRingTime', 'UFCFightNumber', 'Normal_RingTime', 'subs_landed', 'lost_by_sub',
    'five_rounds', 'kd_cumsum', 'kds_received_cumsum', 'sig_strike_attempts_cumsum', 'sig_strike_landed_cumsum',
    'sig_strikes_avoided_cumsum', 'sig_strikes_received_cumsum', 'strike_attempts_cumsum', 'strike_landed_cumsum',
    'strikes_avoided_cumsum', 'strikes_received_cumsum', 'sub_attempts_cumsum', 'subs_landed_cumsum',
    'lost_by_sub_cumsum', 'td_attempts_cumsum', 'td_landed_cumsum', 'tds_defended_cumsum', 'tds_received_cumsum',
    'kd_cumsum_normalized', 'kds_received_cumsum_normalized', 'sig_strike_attempts_cumsum_normalized',
    'sig_strike_landed_cumsum_normalized', 'sig_strikes_avoided_cumsum_normalized', 'sig_strikes_received_cumsum_normalized',
    'strike_attempts_cumsum_normalized', 'strike_landed_cumsum_normalized', 'strikes_avoided_cumsum_normalized',
    'strikes_received_cumsum_normalized', 'sub_attempts_cumsum_normalized', 'subs_landed_cumsum_normalized',
    'lost_by_sub_cumsum_normalized', 'td_attempts_cumsum_normalized', 'td_landed_cumsum_normalized',
    'tds_defended_cumsum_normalized', 'tds_received_cumsum_normalized', 'sig_strike_pct', 'sig_strike_avoided_pct',
    'strike_pct', 'strike_avoided_pct', 'subs_pct', 'td_pct', 'td_avoided_pct', 'sig_strike_attempts_cumsum_normalized_diff',
    'sig_strike_landed_cumsum_normalized_diff', 'strikes_avoided_cumsum_normalized_diff', 'strikes_received_cumsum_normalized_diff',
    'sig_strike_pct_diffs', 'strike_pct_diffs', 'takedown_pct_diffs', 'sig_strike_norm_diffs', 'ConsecutiveLosses',
    'UFCFightNumber2', 'elo', 'EloDiff', 'EloSinceLastFight', 'Age', 'AgeDiff', 'SinceLastFight', 'KnockedOut', 'KnockOuts',
    'kd', 'kds_received', 'sig_reg_mixture', 'sig_reg_percent', 'sig_strike_attempts', 'sig_strike_landed', 'sig_strike_percent',
    'sig_strikes_avoided', 'sig_strikes_received', 'strike_attempts', 'strike_landed', 'strike_percent', 'strikes_avoided',
    'strikes_received', 'sub_attempts', 'td_attempts', 'td_landed', 'td_percent', 'tds_defended', 'tds_received'
    ]

    df_diff = fill_and_adjust_column(df_diff, 'sig_strike_attempts_cumsum_normalized', 'fight_pk')
    df_diff = fill_and_adjust_column(df_diff, 'sig_strike_landed_cumsum_normalized', 'fight_pk')
    df_diff = fill_and_adjust_column(df_diff, 'strikes_avoided_cumsum_normalized', 'fight_pk')
    df_diff = fill_and_adjust_column(df_diff, 'strikes_received_cumsum_normalized', 'fight_pk')
    df_diff = calculate_stat_diff(df_diff, 'sig_strike_pct', 'sig_strike_avoided_pct', 'fight_pk', 'sig_strike_pct_diffs')
    df_diff = calculate_stat_diff(df_diff, 'strike_pct', 'strike_avoided_pct', 'fight_pk', 'strike_pct_diffs')
    df_diff = calculate_stat_diff(df_diff, 'td_pct', 'td_avoided_pct', 'fight_pk', 'takedown_pct_diffs')
    df_diff = calculate_stat_diff(df_diff, 'sig_strike_landed_cumsum_normalized', 'sig_strikes_avoided_cumsum_normalized',
                                'fight_pk', 'sig_strike_norm_diffs')
    
    df_diff.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    preds = model.predict_proba(df_diff[columns_to_include])[:, 1]
    
    columns_to_include_fid = columns_to_include + ['fid']
    
    final = df_diff[columns_to_include_fid]
    fin_df = final
    fin_df = fin_df.set_index('fid')
    fin_df['Pred'] = preds
    
    st.write(fin_df)

    shap_cols = ['pytime', 'RingTime', 'TotalRingTime', 'UFCFightNumber',
       'Normal_RingTime', 'subs_landed', 'lost_by_sub', 'five_rounds',
       'kd_cumsum', 'kds_received_cumsum', 'sig_strike_attempts_cumsum',
       'sig_strike_landed_cumsum', 'sig_strikes_avoided_cumsum',
       'sig_strikes_received_cumsum', 'strike_attempts_cumsum',
       'strike_landed_cumsum', 'strikes_avoided_cumsum',
       'strikes_received_cumsum', 'sub_attempts_cumsum', 'subs_landed_cumsum',
       'lost_by_sub_cumsum', 'td_attempts_cumsum', 'td_landed_cumsum',
       'tds_defended_cumsum', 'tds_received_cumsum', 'kd_cumsum_normalized',
       'kds_received_cumsum_normalized',
       'sig_strike_attempts_cumsum_normalized',
       'sig_strike_landed_cumsum_normalized',
       'sig_strikes_avoided_cumsum_normalized',
       'sig_strikes_received_cumsum_normalized',
       'strike_attempts_cumsum_normalized', 'strike_landed_cumsum_normalized',
       'strikes_avoided_cumsum_normalized',
       'strikes_received_cumsum_normalized', 'sub_attempts_cumsum_normalized',
       'subs_landed_cumsum_normalized', 'lost_by_sub_cumsum_normalized',
       'td_attempts_cumsum_normalized', 'td_landed_cumsum_normalized',
       'tds_defended_cumsum_normalized', 'tds_received_cumsum_normalized',
       'sig_strike_pct', 'sig_strike_avoided_pct', 'strike_pct',
       'strike_avoided_pct', 'subs_pct', 'td_pct', 'td_avoided_pct',
       'sig_strike_attempts_cumsum_normalized_diff',
       'sig_strike_landed_cumsum_normalized_diff',
       'strikes_avoided_cumsum_normalized_diff',
       'strikes_received_cumsum_normalized_diff', 'sig_strike_pct_diffs',
       'strike_pct_diffs', 'takedown_pct_diffs', 'sig_strike_norm_diffs',
       'ConsecutiveLosses', 'UFCFightNumber2', 'elo', 'EloDiff',
       'EloSinceLastFight', 'Age', 'AgeDiff', 'SinceLastFight', 'KnockedOut',
       'KnockOuts', 'kd', 'kds_received', 'sig_reg_mixture', 'sig_reg_percent',
       'sig_strike_attempts', 'sig_strike_landed', 'sig_strike_percent',
       'sig_strikes_avoided', 'sig_strikes_received', 'strike_attempts',
       'strike_landed', 'strike_percent', 'strikes_avoided',
       'strikes_received', 'sub_attempts', 'td_attempts', 'td_landed',
       'td_percent', 'tds_defended', 'tds_received']
    
    with open('explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)

    with open('shap_values.pkl', 'rb') as f:
        shap_values = pickle.load(f)
    
        #shap_values = explainer.shap_values(final[shap_cols])

    
    for i in range(len(final)):
        st.write(f"SHAP values for {final.fid.iloc[i]}")
        shap_values_for_fighter = explainer.shap_values(final[shap_cols])[i]

        # Create a SHAP force plot
        fig, ax = plt.subplots()
        shap.force_plot(explainer.expected_value, shap_values_for_fighter, final[shap_cols].iloc[i].round(2), matplotlib=True, show=False)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img_data = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        
        # Display the SHAP force plot in Streamlit
        st.markdown(f"![SHAP Force Plot](data:image/png;base64,{img_data})")
