import pickle
import cPickle
from Layers.RNNHiddenLayer import *
from Layers.SoftmaxLayer import *

class RNN:
    def __init__(self,
                 rng,
                 input,
                 numIn,
                 numHidden,
                 truncate,
                 useSoftmax=False,
                 activation = T.tanh):
        # Set parameters
        self.Rng = rng
        self.Input = input
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.Truncate = truncate
        self.UseSoftmax = useSoftmax
        self.Activation = activation
        self.Layers = []

        # Save shared parameters
        self.Whh = None
        self.Wx = None
        self.Wy = None

        # Create RNN model
        for layerId in range(self.Truncate):
            hiddenLayer = None
            if layerId == 0:
                hiddenLayer = RNNHiddenLayer(
                    rng = self.Rng,
                    numIn = self.NumIn,
                    numHidden = self.NumHidden,
                    activation = self.Activation
                )
                self.Whh = hiddenLayer.Whh
                self.Wx = hiddenLayer.Wx
                self.Wy = hiddenLayer.Wy
            else:
                hiddenLayer = RNNHiddenLayer(
                    rng=self.Rng,
                    numIn=self.NumIn,
                    numHidden=self.NumHidden,
                    activation=self.Activation,
                    whh = self.Whh,
                    wx = self.Wx,
                    wy = self.Wy
                )
            self.Layers += hiddenLayer

        # Update parameters
        params = [layer.Params for layer in self.Layers]
        grads = T.grad(cost)


    def LoadModel(self, fileName):
        file = open(fileName, 'r')
        self.Whh.set_value(cPickle.load(file), borrow = True)
        self.Whh.set_value(cPickle.load(file), borrow=True)
        self.Whh.set_value(cPickle.load(file), borrow=True)
        file.close()

    def SaveModel(self, fileName):
        file = open(fileName, 'wb')
        pickle.dump(self.Whh.get_value(borrow = True), file, -1)
        pickle.dump(self.Wx.get_value(borrow = True), file, -1)
        pickle.dump(self.Wy.get_value(borrow = True), file, -1)
        file.close()

    def Output(self):
        output = self.Layers[-1].Output
        if self.UseSoftmax is False:
            return output
        else:
            softmaxLayer = SoftmaxLayer(
                input = output
            )
            return softmaxLayer.Output()

    def Cost(self):
        return 0


