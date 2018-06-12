from keras.utils import plot_model
import keras
import pydot

model = keras.models.load_model("model.h5")

plot_model(model, to_file='model.png', show_shapes=True)