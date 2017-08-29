import sys 
import matplotlib.pyplot as plt
import numpy as np
from utils import set_trace

# Read training log
history = np.genfromtxt(sys.argv[-1], names=True, delimiter=",")

# Summarize history for accuracy
plt.plot(history['mean_absolute_error'])
plt.plot(history['val_mean_absolute_error'])
plt.title('Model mean absolute error (MAE)')
plt.ylabel('MAE')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()