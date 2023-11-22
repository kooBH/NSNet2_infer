import librosa as rs
import numpy as np
from TRANet import TRANetEnhancer
from NSNet2 import NSNet2_infer
from tqdm.auto import tqdm
import time

n_iter = 100


#  Load test file
x = rs.load("input.wav",sr=16000)[0]
n_sample = len(x)
duration = n_sample/16000
x2 = np.expand_dims(x,0)


#  Load models
m1 = TRANetEnhancer("mpANC_v99.onnx")
m2 = NSNet2_infer()

# estimatte RTF
tic = time.time()
for i in tqdm(range(n_iter)) : 
    y = m2(x)
toc = time.time()
print("RTF NSNet2 : {}".format((toc-tic)/n_iter/duration))


tic = time.time()
for i in tqdm(range(n_iter)) : 
    y = m1(x2)
toc = time.time()
print("RTF TRANet : {}".format((toc-tic)/n_iter/duration))

