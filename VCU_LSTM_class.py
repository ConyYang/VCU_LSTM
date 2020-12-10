from tensorflow.keras import regularizers, Model
from tensorflow.keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
import pandas as pd
import numpy as np
from custom_round import round_decimals_up, round_decimals_down


class VCU_LSTM(object):
    def __init__(self, dataframe):
        self.train_len = int(len(dataframe) * (3 / 4))
        self.test_len = int(len(dataframe) * (1 / 4))
        self.train_df = dataframe[:self.train_len]
        self.test_df = dataframe[-self.test_len:]

        self.X_train = self.train_df.to_numpy().reshape(dataframe[:self.train_len].shape[0],
                                                        1, dataframe[-self.test_len:].shape[1])
        self.X_test = self.test_df.to_numpy().reshape(dataframe[-self.test_len:].shape[0],
                                                      1, dataframe[-self.test_len:].shape[1])
        self.scored_train = pd.DataFrame(index=self.train_df.index)
        self.scored_test = pd.DataFrame(index=self.test_df.index)

        self.model = None
        self.threshold = None
        self.anomaly_count = 0
        self.accuracy = 0.0

    def create_autoencoder_model(self):
        X = self.X_train
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        Layer_1 = LSTM(16, activation='relu',
                       return_sequences=True,
                       kernel_regularizer=regularizers.l2(0.00))(inputs)
        Layer_2 = LSTM(4, activation='relu', return_sequences=False)(Layer_1)
        Layer_3 = RepeatVector(X.shape[1])(Layer_2)
        Layer_4 = LSTM(4, activation='relu', return_sequences=True)(Layer_3)
        Layer_5 = LSTM(16, activation='relu', return_sequences=True)(Layer_4)
        outputs = TimeDistributed(Dense(X.shape[2]))(Layer_5)
        self.model = Model(inputs=inputs, outputs=outputs)

    def model_compile(self, epochs=10):
        self.model.compile(optimizer='adam', loss='mae')
        epochs = epochs
        batch_size = 10
        history = self.model.fit(self.X_train, self.X_train, epochs=epochs,
                            batch_size=batch_size, validation_split=0.1, verbose=0)

    def model_predict_train(self):
        X_pred = self.model.predict(self.X_train)
        X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
        X_pred = pd.DataFrame(X_pred, columns=self.train_df.columns)
        X_pred.index = self.train_df.index
        X_pred = X_pred.to_numpy()


        X_train_plt = self.X_train.reshape(self.X_train.shape[0], self.X_train.shape[2])

        self.scored_train['Loss_MAE'] = np.mean(np.abs(X_pred - X_train_plt), axis=1)
        self.threshold = round_decimals_up(np.max(self.scored_train['Loss_MAE']), 4)

    def model_predict_test(self):
        X_pred_test = self.model.predict(self.X_test)
        X_pred_test = X_pred_test.reshape(X_pred_test.shape[0], X_pred_test.shape[2])
        X_pred_test = pd.DataFrame(X_pred_test, columns=self.test_df.columns)
        X_pred_test.index = self.test_df.index
        X_pred_test = X_pred_test.to_numpy()

        self.scored_test = pd.DataFrame(index=self.test_df.index)
        X_test_plt = self.X_test.reshape(self.X_test.shape[0], self.X_test.shape[2])

        self.scored_test['Loss_MAE'] = np.mean(np.abs(X_pred_test - X_test_plt), axis=1)
        print(self.scored_test.head())
        self.scored_test['Threshold'] = self.threshold
        self.scored_test['Anomaly'] = self.scored_test['Loss_MAE'] > self.scored_test['Threshold']
        self.scored_test.head()

        self.anomaly_count = self.scored_test['Anomaly'].value_counts()
        self.accuracy = self.anomaly_count/len(self.X_test)

    def cal_metrics(self, figname):
        """
        calculate the same metrics for the training set and merge all data into a single dataframe for plotting
        :param scored_train: {DataFrame: (1500, 3)}
        :param scored_test: {DataFrame: (500, 3)}
        :return: None
        """
        self.scored_train['Threshold'] = self.threshold
        self.scored_train['Anomaly'] = self.scored_train['Loss_MAE'] > self.scored_train['Threshold']
        concat_score = pd.concat([self.scored_train, self.scored_test])
        figname = figname
        concat_score.plot(logy=True, figsize=(16, 9), ylim=[round_decimals_down(self.threshold,3), round_decimals_up(self.threshold,3)],
                          color=['blue', 'red']).get_figure().savefig(figname)


if __name__ == '__main__':
    path = 'scaled_data/VCU_P_5SignalData_scaled.csv'
    dataframe = pd.read_csv(path)
    LSTM1 = VCU_LSTM(dataframe)

    LSTM1.create_autoencoder_model()
    LSTM1.model_compile(epochs=20)
    LSTM1.model_predict_train()
    LSTM1.model_predict_test()
    LSTM1.cal_metrics(figname='threshold' + '')

    with open("demofile2.txt", "a") as f:
        f.truncate(0)
        f.write(str(LSTM1.threshold))
        f.write('\n')
        f.write(str(LSTM1.scored_train.head()))
        f.write('\n')
        f.write(str(LSTM1.scored_test.head()))
        f.write('\n')
        f.write(str(LSTM1.anomaly_count)[:12])
        f.write('\n')
        f.write(str(LSTM1.accuracy)[:12])