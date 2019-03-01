import chainer
from chainer import Variable
import chainer.links as L
import chainer.functions as F
import numpy as np
import math
import cv2
from chainer.links.caffe import CaffeFunction
import chainerrl
from chainerrl.agents import a3c

class MyFcn_trained(chainer.Chain, a3c.A3CModel):

    def __init__(self, n_actions):
        super(MyFcn_trained, self).__init__(
            conv1=L.Convolution3D( 1, 16, 3, stride=1, pad=1, nobias=False, initialW=None, initial_bias=None),
            diconv2=L.Convolution3D(16, 16, 3, dilate=2, stride=1, pad=2),
            # diconv2=DilatedConvBlock(2, None, None),
            diconv3=L.Convolution3D(16, 16, 3, dilate=3, stride=1, pad=3),
            # diconv3=DilatedConvBlock(3, None, None),
            diconv4=L.Convolution3D(16, 16, 3, dilate=4, stride=1, pad=4),
            # diconv4=DilatedConvBlock(4, None, None),
            diconv5_pi=L.Convolution3D(16, 16, 3, dilate=3, stride=1, pad=3),
            # diconv5_pi=DilatedConvBlock(3, None, None),
            diconv6_pi=L.Convolution3D(16, 16, 3, dilate=2, stride=1, pad=2),
            # diconv6_pi=DilatedConvBlock(2, None, None),
            conv7_Wz=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_Uz=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_Wr=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_Ur=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_W=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv7_U=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=None),
            conv8_pi=chainerrl.policies.SoftmaxPolicy(L.Convolution3D( 16, n_actions, 3, stride=1, pad=1, nobias=False, initialW=None)),
            diconv5_V=L.Convolution3D(16, 16, 3, dilate=3, stride=1, pad=3),
            # diconv5_V=DilatedConvBlock(3, None, None),
            diconv6_V=L.Convolution3D(16, 16, 3, dilate=2, stride=1, pad=2),
            # diconv6_V=DilatedConvBlock(2, None, None),
            conv7_V=L.Convolution3D( 16, 1, 3, stride=1, pad=1, nobias=False, initialW=None, initial_bias=None),
        )


class DilatedConvBlock(chainer.Chain):

    def __init__(self, d_factor, weight, bias):
        super(DilatedConvBlock, self).__init__(
            diconv=L.DilatedConvolution2D( in_channels=16, out_channels=16, ksize=3, stride=1, pad=d_factor, dilate=d_factor, nobias=False, initialW=weight, initial_bias=bias),
            #bn=L.BatchNormalization(16)
        )

        self.train = True

    def __call__(self, x):
        h = F.relu(self.diconv(x))
        #h = F.relu(self.bn(self.diconv(x)))
        return h


class MyFcn(chainer.Chain, a3c.A3CModel):

    def __init__(self, n_actions):
        w = chainer.initializers.HeNormal()
        # wI = np.zeros((1,1,17,17,17))
        wI = np.zeros((1,1,33,33,33))
        # wI[:,:,8,8,8] = 1
        wI[:,:,16,16,16] = 1
        net = MyFcn_trained(n_actions)
        # chainer.serializers.load_npz('../denoise_with_convGRU/model/pretrained_15.npz', net)
        super(MyFcn, self).__init__(
            conv1=L.Convolution3D( 1, 16, 3, stride=1, pad=1, nobias=False, initialW=net.conv1.W.data, initial_bias=net.conv1.b.data),
            diconv2=L.Convolution3D(16, 16, 3, dilate=2, pad=2, initialW=net.diconv2.W.data, initial_bias=net.diconv2.b.data),
            # diconv2=DilatedConvBlock(2, net.diconv2.diconv.W.data, net.diconv2.diconv.b.data),
            diconv3=L.Convolution3D(16, 16, 3, dilate=3, stride=1, pad=3, initialW=net.diconv3.W.data, initial_bias=net.diconv3.b.data),
            # diconv3=DilatedConvBlock(3, net.diconv3.diconv.W.data, net.diconv3.diconv.b.data),
            diconv4=L.Convolution3D(16, 16, 3, dilate=4, stride=1, pad=4, initialW=net.diconv4.W.data, initial_bias=net.diconv4.b.data),
            # diconv4=DilatedConvBlock(4, net.diconv4.diconv.W.data, net.diconv4.diconv.b.data),
            diconv5_pi=L.Convolution3D(16, 16, 3, dilate=3, stride=1, pad=3, initialW=net.diconv5_pi.W.data, initial_bias=net.diconv5_pi.b.data),
            # diconv5_pi=DilatedConvBlock(3, net.diconv5_pi.diconv.W.data, net.diconv5_pi.diconv.b.data),
            diconv6_pi=L.Convolution3D(16, 16, 3, dilate=2, stride=1, pad=2, initialW=net.diconv6_pi.W.data, initial_bias=net.diconv6_pi.b.data),
            # diconv6_pi=DilatedConvBlock(2, net.diconv6_pi.diconv.W.data, net.diconv6_pi.diconv.b.data),
            conv7_Wz=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_Wz.W.data),
            conv7_Uz=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_Uz.W.data),
            conv7_Wr=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_Wr.W.data),
            conv7_Ur=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_Ur.W.data),
            conv7_W=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_W.W.data),
            conv7_U=L.Convolution3D( 16, 16, 3, stride=1, pad=1, nobias=True, initialW=net.conv7_U.W.data),
            conv8_pi=chainerrl.policies.SoftmaxPolicy(L.Convolution3D( 16, n_actions, 3, stride=1, pad=1, nobias=False, initialW=net.conv8_pi.model.W.data, initial_bias=net.conv8_pi.model.b.data)),
            diconv5_V=L.Convolution3D(16, 16, 3, dilate=3, stride=1, pad=3, initialW=net.diconv5_V.W.data, initial_bias=net.diconv5_V.b.data),
            # diconv5_V=DilatedConvBlock(3, net.diconv5_V.diconv.W.data, net.diconv5_V.diconv.b.data),
            diconv6_V=L.Convolution3D(16, 16, 3, dilate=2, stride=1, pad=2, initialW=net.diconv6_V.W.data, initial_bias=net.diconv6_V.b.data),
            # diconv6_V=DilatedConvBlock(2, net.diconv6_V.diconv.W.data, net.diconv6_V.diconv.b.data),
            conv7_V=L.Convolution3D( 16, 1, 3, stride=1, pad=1, nobias=False, initialW=net.conv7_V.W.data, initial_bias=net.conv7_V.b.data),
            # conv_R=L.Convolution2D( 1, 1, 17, stride=1, pad=8, nobias=True, initialW=wI),
            conv_R=L.Convolution3D( 1, 1, 33, stride=1, pad=16, nobias=True, initialW=wI),
        )
        self.train = True

    def pi_and_v(self, x):
        # print(x.shape)
        # print("****")
        h = F.relu(self.conv1(x[:,0:1,:,:,:]))
        # print('h1',h.shape)
        h = self.diconv2(h)
        # print('h2',h.shape)
        h = self.diconv3(h)
        # print('h3',h.shape)
        h = self.diconv4(h)
        # print('h4',h.shape)
        # print('h',h.shape)

        h_pi = self.diconv5_pi(h)
        # print('h_pi',h_pi.shape)
        x_t = self.diconv6_pi(h_pi)
        # print('x_t',x_t.shape)
        h_t1 = x[:,-16:,:,:]
        # print('h_t1',h_t1.shape)
        z_t = F.sigmoid(self.conv7_Wz(x_t)+self.conv7_Uz(h_t1))
        r_t = F.sigmoid(self.conv7_Wr(x_t)+self.conv7_Ur(h_t1))
        h_tilde_t = F.tanh(self.conv7_W(x_t)+self.conv7_U(r_t*h_t1))
        h_t = (1-z_t)*h_t1+z_t*h_tilde_t
        pout = self.conv8_pi(h_t)

        h_V = self.diconv5_V(h)
        h_V = self.diconv6_V(h_V)
        vout = self.conv7_V(h_V)

        return pout, vout, h_t

    def conv_smooth(self, x):

        x = self.conv_R(x)

        return x
