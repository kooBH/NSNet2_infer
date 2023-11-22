import librosa as rs
import soundfile as sf

from enhance_onnx import NSnet2Enhancer

class NSNet2_infer :
    def __init__(self):

        self.cfg = {
            'winlen'   : 0.02,
            'hopfrac'  : 0.5,
            'fs'       : 16000,
            'mingain'  : -80,
            'feattype' : 'LogPow',
            'nfft'     : 320
        }

        self.fs = self.cfg['fs']

        self.enhancer = NSnet2Enhancer(modelfile="nsnet2-20ms-baseline.onnx", cfg=self.cfg)

    def __call__(self,x):

        outSig = self.enhancer(x,self.fs)

        return outSig
