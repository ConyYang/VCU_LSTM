import pandas as pd
import numpy as np
import os


def distribution(np_arr_signal):
    min_val = min(np_arr_signal)
    max_val = max(np_arr_signal)
    variance = np.var(np_arr_signal)
    mean = np.mean(np_arr_signal)
    standard_deviation = np.std(np_arr_signal)
    median = np.median(np_arr_signal)

    return min_val, max_val, variance, mean, standard_deviation, median


if __name__ == '__main__':
    paths = os.listdir('Decimal_Combine_Signal')
    paths.sort()
    array = []

    for message in paths:

        df = pd.read_csv(('Decimal_Combine_Signal/' + message))
        for signal in df.columns[3:]:
            min_val, max_val, variance, mean, standard_deviation, median = distribution(df[signal])

            array.append([message, signal[2:], min_val, max_val,
                          variance, mean, standard_deviation, median])

        result_df = pd.DataFrame(np.array(array),
                                 columns=['Message Name', 'Signal Name',
                                          'min_val', 'max_val', 'variance', 'mean',
                                          'standard_deviation', 'median'])
    result_df.to_csv('data_distribution.csv')
