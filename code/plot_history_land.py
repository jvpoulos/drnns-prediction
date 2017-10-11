import sys 
import matplotlib.pyplot as plt
import numpy as np
from utils import set_trace

# Read training log
history = np.genfromtxt(sys.argv[-2], names=True, delimiter=",")

# Summarize history for accuracy
plt.plot(history['mean_absolute_error'])
plt.plot(history['val_mean_absolute_error'])
plt.title(sys.argv[-1])
plt.ylabel('Mean absolute error (MAE)')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()