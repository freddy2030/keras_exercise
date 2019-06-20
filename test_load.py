from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
import numpy as np

x_test = np.array([[0,0]])

model = load_model('simple.h5')

print("test: ", model.predict(x_test))
# print(model.get_weights()[1])
for w in model.get_weights():
    print(w)
