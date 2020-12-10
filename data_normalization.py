import pandas as pd
from sklearn import preprocessing

df = pd.read_csv('Decimal_Combine_Signal/VCU_P_7SignalData.csv')
# cols_to_scale = ['01VCU_Tx_trqMCUFL',
#                  '06VCU_Tx_stMCUFLReq',
#                  '10VCU_Tx_ctRolling5',
#                   '11VCU_Tx_numChkS5']

cols_to_scale = ['01VCU_Tx_trqMCURL',
                 '06VCU_Tx_stMCURLReq',
                 '10VCU_Tx_ctRolling7',
                  '11VCU_Tx_numChkS7']

df_extract = df[cols_to_scale]
scaler = preprocessing.MinMaxScaler()

df_extract[cols_to_scale] = scaler.fit_transform(df_extract[cols_to_scale])

df_extract.to_csv('scaled_data/VCU_P_7SignalData_scaled.csv')

