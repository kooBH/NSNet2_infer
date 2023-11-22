import torch
import numpy as np
import onnxruntime as ort


class TRANetEnhancer(object):
    def __init__(self,path_model):

        self.n_fft = 512
        self.n_hop = 128
        self.sr = 16000

        """load onnx model"""
        self.ort = ort.InferenceSession(path_model)

        print("len inputs : {}".format(len(self.ort.get_inputs())))

        for i in range(len(self.ort.get_inputs())) : 
            print("{} | {} : {}".format(i,self.ort.get_inputs()[i].name,self.ort.get_inputs()[i].shape))




    def __call__(self,x):

        X = torch.stft(torch.from_numpy(x), n_fft = self.n_fft, window=torch.hann_window(self.n_fft),return_complex=False)
        X = X.float().numpy()

        L = X.shape[2]
        onnx_inputs = None

        Y = torch.zeros(X.shape)

        for i in range(L):
            if onnx_inputs is None:
                onnx_inputs = {
                    self.ort.get_inputs()[0].name: X[:,:,i:i+1,:],
                    self.ort.get_inputs()[1].name: np.zeros((1,17,64),np.float32),
                    self.ort.get_inputs()[2].name: np.zeros((1,257,2,2),np.float32)
                            }
            else:
                onnx_inputs[self.ort.get_inputs()[0].name] = X[:,:,i:i+1,:]

            out = self.ort.run(None, onnx_inputs)
            onnx_inputs[self.ort.get_inputs()[1].name] = out[1]
            onnx_inputs[self.ort.get_inputs()[2].name] = out[2]

            Y[:,:,i:i+1,:] = torch.from_numpy(out[0])


        y = torch.istft(Y[:,:,:,0] + 1j*Y[:,:,:,1], self.n_fft, self.n_hop, self.n_fft, torch.hann_window(self.n_fft),return_complex=False)


        return y


if __name__ == "__main__":

    x = torch.randn(1,16000)
    m = TRANetEnhancer("mpANC_v99.onnx")
    y = m(x.numpy())

