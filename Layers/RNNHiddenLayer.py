import theano
import theano.tensor as T
import numpy

class RNNHiddenLayer:


    def __init__(self,
                 rng,
                 numIn,
                 numHidden,
                 whh = None,
                 wx = None,
                 wy = None,
                 skm1 = None,
                 activation = T.tanh):
        # Set parameters
        self.Rng = rng
        self.NumIn = numIn
        self.NumHidden = numHidden
        self.Activation = activation

        if whh is None:
            self.Whh = theano.shared(
                numpy.asarray(rng.uniform(
                        low = -1.0,
                        high = 1.0,
                        size = (self.NumHidden, self.NumHidden)
                    ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )
        else:
            self.Whh = whh;

        if wx is None:
            self.Wx = theano.shared(
                numpy.asarray(rng.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.NumIn, self.NumHidden)
                ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )
        else:
            self.Wx = wx

        if wy is None:
            self.Wy = theano.shared(
                numpy.asarray(rng.uniform(
                    low = -1.0,
                    high = 1.0,
                    size = (self.NumHidden, self.NumIn)
                ),
                    dtype = theano.config.floatX
                ),
                borrow = True
            )
        else:
            self.Wy = wy

        if skm1 is None:
            self.Skm1 = theano.shared(
                numpy.zeros(
                    shape = (self.NumHidden),
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            self.Skm1 = skm1

    def Output(self):
        return 0

    def Params(self):
        return [self.Whh,
                self.Wx,
                self.Wy]

    def Y(self, Xk):
        s = self.S(Xk)
        y = s * self.Wy
        if self.Activation is None:
            return y
        else:
            return self.Activation(y)

    def S(self, Xk):
        s = T.dot(self.Skm1, self.hh) + T.dot(self.Wx, Xk)
        if self.Activation is None:
            return s
        else:
            return self.Activation(s)