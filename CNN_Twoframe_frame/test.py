import numpy as np
import tensorflow as tf
import pandas as pd
from matplotlib import pyplot as plt

SNR = [1,2,3,4]
SNR = np.array(SNR)


SNR = (SNR-2)*6
a=[0.9547999998368323, 0.8395999993383885, 0.7076000016927719, 0.6418999993801117]
pd.DataFrame(data=a, index=SNR).plot()

plt.title('CNN FS')
plt.xlabel('SNR')
plt.ylabel('Error probability of FS')
plt.grid()
plt.savefig('F:\Python_project\CNN_Twoframe_frame\CNN_FS.pdf')
plt.show()

# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(c))

