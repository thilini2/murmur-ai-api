import tensorflow as tf

class PredictionModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        print(tf.__version__)
        physical_devices = tf.config.list_physical_devices('GPU')
        print("Available GPUs:", physical_devices)
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)


    def predict(self, input_data):
        return self.model.predict(input_data)