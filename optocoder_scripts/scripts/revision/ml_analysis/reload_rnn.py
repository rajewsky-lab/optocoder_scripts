from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from tensorflow import keras
from keras import backend as K
from scipy import stats
import kerastuner as kt

class RNNHyperModel(kt.HyperModel):

    def __init__(self, num_cycles):
        self.num_cycles = num_cycles

    def build(self, hp):
        model = keras.models.Sequential()
        model.add(keras.layers.Bidirectional(
            keras.layers.LSTM(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu',
                                return_sequences=True), input_shape=(self.num_cycles, 4)))
        model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))

        model.add(keras.layers.Dense(4, activation='softmax'))

        model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])),
                        loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
        
def get_rnn_model(num_cycles, hp):
    model = keras.models.Sequential()
    model.add(keras.layers.Bidirectional(
        keras.layers.LSTM(hp.Int('units', min_value=32, max_value=512, step=32), activation='relu',
                            return_sequences=True), input_shape=(num_cycles, 4)))
    model.add(keras.layers.Dropout(hp.Float('dropout', 0, 0.5, step=0.1)))

    model.add(keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])),
                    loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model