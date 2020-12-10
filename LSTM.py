import pandas as pd
from VCU_LSTM_class import VCU_LSTM
import os


def create_model(file, epochs=5):
    dataframe = pd.read_csv(file)
    LSTM_model = VCU_LSTM(dataframe)

    LSTM_model.create_autoencoder_model()
    LSTM_model.model_compile(epochs=epochs)
    LSTM_model.model_predict_train()
    LSTM_model.model_predict_test()
    return LSTM_model


def write_results(LSTM_model, f, figname):
    LSTM_model.cal_metrics(figname=figname)
    f.write(str(LSTM_model.threshold))
    f.write('\n')
    f.write(str(LSTM_model.scored_train.head()))
    f.write('\n')
    f.write(str(LSTM_model.scored_test.head()))
    f.write('\n')
    f.write(str(LSTM_model.anomaly_count)[:12])
    f.write('\n')
    f.write(str(LSTM_model.accuracy)[:12])


if __name__ == '__main__':
    root_dirs = os.listdir('scaled_data')

    for directory in root_dirs:
            EPOCHS = 100
            write_txt_path = 'LSTM_result/' + directory + '.txt'

            with open(write_txt_path, "a") as f:
                f.truncate(0)
                f.write('\n')
                f.write('\n')
                csv_full_path = 'scaled_data' + '/' + directory
                print(csv_full_path)
                LSTM_model = create_model(csv_full_path, epochs=EPOCHS)
                figname = 'LSTM_result/' + directory[:-4]+ '_threshold.png'
                write_results(LSTM_model,f,figname)
                f.write('\n')

