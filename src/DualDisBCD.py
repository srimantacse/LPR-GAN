from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Reshape, core, Dropout, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
from keras.utils.vis_utils import plot_model as plot
from keras.optimizers import SGD
from keras.optimizers import *
from keras.layers import *    
from keras.initializers import RandomNormal, GlorotNormal
import cv2
import numpy as np
from keras.preprocessing.image import array_to_img
import tensorflow as tf
import os

from img_utils import get_batch

def accw(y_true, y_pred):
      y_true = K.clip(y_true, -1, 1)
      y_pred = K.clip(y_pred, -1, 1)
      return K.mean(K.equal(y_true, K.round(y_pred)))

def mssim(y_true, y_pred):
  costs = 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))
  return costs

def wloss(y_true,y_pred):
    y_true = K.clip(y_true, -1, 1)
    y_pred = K.clip(y_pred, -1, 1)
    return -K.mean(y_true*y_pred)
  
def PSNR(y_true, y_pred):
            return -10. * K.log(K.mean(K.square(y_pred - y_true))) / K.log(10.)

class DualDisScaledGAN:
    def __init__(self, image_shape_gen=(128, 128, 3), image_shape_dis=(128, 128, 1), patch_size=(8, 8), bs = 2,
                 gen_loss=['binary_crossentropy'], gen_loss_wght=[100]):
        super(DualDisScaledGAN, self).__init__()
        
        self.image_shape_gen = image_shape_gen
        self.image_shape_dis = image_shape_dis
        self.patch_size      = patch_size
        self.batch_size      = bs
        
        self.d_model1        = self._define_discriminator1()
        self.d_model2        = self._define_discriminator2()
        self.g_model         = self._define_generator(gen_loss, gen_loss_wght)
        self.gan_model       = self._define_gan(gen_loss, gen_loss_wght)
        self.X_realA_List    = []
        self.X_realB_List    = []
        self.iteartion_per_epoch = 0
        
    def feed_data(self, path_X, path_Y):
        X1 = sorted(os.listdir(path_X))
        Y1 = sorted(os.listdir(path_Y))
        X1 = [x for x in X1 if "blur" not in x]
        Y1 = [y for y in Y1 if "blur" not in y]
        X = np.reshape(X1, (-1, self.batch_size))
        Y = np.reshape(Y1, (-1, self.batch_size))
        self.iteartion_per_epoch = len(X)

        for batch in range(len(X)):
            X_realA, X_realB = get_batch(X[batch], Y[batch], path_X, path_Y, 
                                         (self.image_shape_gen[1], self.image_shape_gen[0]))
            self.X_realA_List.append(X_realA)
            self.X_realB_List.append(X_realB)
        if (len(self.X_realA_List) != len(self.X_realB_List)):
          print('Error in training image count')
        print('Batch Size:', self.batch_size, 
              'Iteartion Per Epoch:', self.iteartion_per_epoch)
        

    def run(self, model_dir, no_epoch=10, log_print=True):
        bs             = self.batch_size
        patch1, patch2 = self.patch_size

        for epoch in range(no_epoch):
          for batch in range(self.iteartion_per_epoch):

            # Real Input
            X_realA, X_realB = self.X_realA_List[batch], self.X_realB_List[batch]
            y_real_patch     = np.ones((bs, patch1, patch2, 1))   
            y_real           = np.ones((bs, 2))

            # Produce Fake Image through Generator
            X_fakeB      = self.g_model.predict([X_realA, X_realB])
            y_fake_patch = np.zeros((bs, patch1, patch2, 1))
            y_fake       = np.zeros((bs, 2))

           
            ###### Train Discriminator
            # Unfreeze Discriminator
            self.d_model1.trainable = True
            self.d_model2.trainable = True

            # Loss in Dis 1
            d_loss1_patch = self.d_model1.train_on_batch(X_realB, y_real_patch)
            d_loss1       = self.d_model2.train_on_batch(X_realB, y_real)

              # Loss in Dis 2
            d_loss2_patch = self.d_model1.train_on_batch(X_fakeB, y_fake_patch)
            d_loss2       = self.d_model2.train_on_batch(X_fakeB, y_fake)

            ###### Train Generator through GAN
            # Freeze Discriminator
            self.d_model1.trainable = False
            self.d_model2.trainable = False

              # Update the Generator through GAN train
            g_loss = self.gan_model.train_on_batch(X_realA, [y_real_patch, y_real, X_realB, X_realB])

              # Summarize performance
            if log_print:
              print('>> epoch %02d batch %02d d1[%.3f] d2[%.3f] g[%.3f] gan[%s]' % 
                    (epoch, batch, d_loss1_patch + d_loss2_patch, d_loss1 + d_loss2, g_loss[3] + g_loss[4], str(g_loss)))
              log_print = True
          
            self.g_model.save(os.path.join(model_dir,  'BCDUnet_gan_gen.h5'))
            self.d_model1.save(os.path.join(model_dir, 'BCDUnet_gan_dis_patch.h5'))
            self.d_model2.save(os.path.join(model_dir, 'BCDUnet_gan_dis_full.h5'))

          if (epoch % 10 == 0):
            print('Saving Model of iteration ' + str(epoch))
            self.g_model.save(os.path.join( model_dir,  'BCDUnet_gan_gen_' + str(epoch) + '.h5'))

            
            
      

    def _define_multiscale(self, inputs):
        # Corse 1 
        conv1  = Conv2D(64, (11, 11), activation='relu', padding='same', 
                        kernel_initializer='he_normal',
                        name="ms_conv_c1")(inputs)
        pool1  = MaxPooling2D(pool_size=(2, 2), name="ms_pool_c1")(conv1)
        up1    = UpSampling2D(name="ms_upsample_c1")(pool1)

        # Corse 2
        conv2  = Conv2D(64, (9, 9), activation='relu', padding='same', 
                        kernel_initializer='he_normal',
                        name="ms_conv_c2")(up1)
        pool2  = MaxPooling2D(pool_size=(2, 2), name="ms_pool_c2")(conv2)
        up2    = UpSampling2D(name="ms_upsample_c2")(pool2)

        # Corse 3
        conv3  = Conv2D(64, (7, 7), activation='relu', padding='same', 
                        kernel_initializer='he_normal',
                        name="ms_conv_c3")(up2)
        pool3  = MaxPooling2D(pool_size=(2, 2), name="ms_pool_c3")(conv3)
        up3    = UpSampling2D(name="ms_upsample_c3")(pool3)



        # Fine 1 
        conv_f1  = Conv2D(64, (7, 7), activation='relu', padding='same', 
                        kernel_initializer='he_normal',
                        name="ms_conv_f1")(inputs)
        pool_f1  = MaxPooling2D(pool_size=(2, 2), name="ms_pool_f1")(conv_f1)
        up_f1    = UpSampling2D(name="ms_upsample_f1")(pool_f1)

        # Fine 2
        concat_up_f1 = concatenate([up_f1, up3], axis=3)
        conv_f2  = Conv2D(64, (5, 5), activation='relu', padding='same', 
                        kernel_initializer='he_normal',
                        name="ms_conv_f2")(concat_up_f1)
        pool_f2  = MaxPooling2D(pool_size=(2, 2), name="ms_pool_f2")(conv_f2)
        up_f2    = UpSampling2D(name="ms_upsample_f2")(pool_f2)

        # Fine 3
        conv_f3  = Conv2D(64, (3, 3), activation='relu', padding='same', 
                        kernel_initializer='he_normal',
                        name="ms_conv_f3")(up_f2)
        pool_f3  = MaxPooling2D(pool_size=(2, 2), name="ms_pool_f3")(conv_f3)
        up_f3    = UpSampling2D(name="ms_upsample_f3")(pool_f3)
        
        return up_f3
      

    def _define_generator(self, loss_in, loss_wghts):
        ######## BCDU_net_D3  
        input_size = self.image_shape_gen   
        N = input_size[0]
        M = input_size[1]
        inputs = Input(input_size)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
      
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        drop3 = Dropout(0.5)(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        # D1
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
        conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4_1 = Dropout(0.5)(conv4_1)
        # D2
        conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
        conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
        conv4_2 = Dropout(0.5)(conv4_2)
        # D3
        merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
        conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
        conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
        drop4_3 = Dropout(0.5)(conv4_3)
        up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
        up6 = BatchNormalization(axis=3)(up6)
        up6 = Activation('relu')(up6)

        x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(M/4), 256))(drop3)
        x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(M/4), 256))(up6)
        merge6  = concatenate([x1,x2], axis = 1) 
        merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
                
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
        up7 = BatchNormalization(axis=3)(up7)
        up7 = Activation('relu')(up7)

        x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(M/2), 128))(conv2)
        x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(M/2), 128))(up7)
        merge7  = concatenate([x1,x2], axis = 1) 
        merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
            
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
        up8 = BatchNormalization(axis=3)(up8)
        up8 = Activation('relu')(up8)

        ###### Multi Scale
        mul1 = self._define_multiscale(inputs)  

        x1 = Reshape(target_shape=(1, N, M, 64))(conv1)
        x2 = Reshape(target_shape=(1, N, M, 64))(up8)
        x3 = Reshape(target_shape=(1, N, M, 64))(mul1)

        merge8 = concatenate([x1,x2,x3], axis = 1) 
        merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', 
                            return_sequences = False, go_backwards = True,
                            kernel_initializer = 'he_normal' )(merge8)    
        
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
        conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

        model = Model(inputs, conv9)

        #opt = Adam(lr=0.0002, beta_1=0.5)
        #model.compile(loss=loss_in, loss_weights=loss_wghts, optimizer=opt)
        return model
        

    def _define_discriminator1(self, loss_in='binary_crossentropy'):
        image_shape = self.image_shape_dis
        init = RandomNormal(stddev=0.02)
        inputs = Input(shape=image_shape)

        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(inputs)
        d = LeakyReLU(alpha=0.2)(d)
        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C256
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # C512
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        # second last output layer
        d = Conv2D(32, (4,4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        # patch output
        print('Patch Size', '(', int(self.patch_size[1]/2), 'x', int(self.patch_size[0]/2), ')')
        d = Conv2D(1, (int(self.patch_size[0]/2),int(self.patch_size[1]/2)), padding='same', kernel_initializer=init)(d)
        patch_out = Activation('sigmoid')(d)

        # define model
        model = Model(inputs, patch_out)
        opt   = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=loss_in, optimizer=opt)
        return model
        
        
    def _define_discriminator2(self, loss_in='binary_crossentropy'):
        image_shape = self.image_shape_dis
        init   = GlorotNormal()
        inputs = Input(shape=image_shape)

        # C64
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(inputs)
        d = LeakyReLU(alpha=0.2)(d)

        # C128
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        # C256
        d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = MaxPooling2D((2, 2))(d)

        # C512
        d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)

        d = Conv2D(32, (4,4), padding='same', kernel_initializer=init)(d)
        d = BatchNormalization()(d)
        d = LeakyReLU(alpha=0.2)(d)
        d = MaxPooling2D((2, 2))(d)

        d = Flatten()(d)

        d = Dense(1024, activation='relu', kernel_initializer='he_uniform')(d)
        d = Dense(256, activation='relu', kernel_initializer='he_uniform')(d)
        d = Dense(64, activation='relu', kernel_initializer='he_uniform')(d)
        out = Dense(2, activation='softmax', kernel_initializer='he_uniform')(d)

        model = Model(inputs, out)
        opt   = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss=loss_in, optimizer=opt)
        
        return model
        
    def define_discriminator3(self, inp_shape, trainable = False):
        
        gamma_init = RandomNormal(stddev=0.02)
        
        inp = Input(shape=inp_shape)
        
        l0 = Conv2D(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp) #b_init is set to none, maybe they are not using bias here, but I am.
        l0 = LeakyReLU(alpha=0.2)(l0)
        
        l1 = Conv2D(64*2, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l0)
        l1 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l1)
        l1 = LeakyReLU(alpha=0.2)(l1)
        
        l2 = Conv2D(64*4, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l1)
        l2 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l2)
        l2 = LeakyReLU(alpha=0.2)(l2)
        
        l3 = Conv2D(64*8, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l2)
        l3 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l3)
        l3 = LeakyReLU(alpha=0.2)(l3)
        
        l4 = Conv2D(64*16, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l3)
        l4 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l4)
        l4 = LeakyReLU(alpha=0.2)(l4)
        
        l5 = Conv2D(64*32, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l4)
        l5 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l5)
        l5 = LeakyReLU(alpha=0.2)(l5)
        
        l6 = Conv2D(64*16, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l5)
        l6 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l6)
        l6 = LeakyReLU(alpha=0.2)(l6)
        
        l7 = Conv2D(64*8, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l6)
        l7 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l7)
        l7 = LeakyReLU(alpha=0.2)(l7)
        #x
        
        l8 = Conv2D(64*2, (1,1), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l7)
        l8 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l8)
        l8 = LeakyReLU(alpha=0.2)(l8)
        
        l9 = Conv2D(64*2, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l8)
        l9 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l9)
        l9 = LeakyReLU(alpha=0.2)(l9)
        
        l10 = Conv2D(64*8, (3,3), strides = (1,1), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l9)
        l10 = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(l10)
        l10 = LeakyReLU(alpha=0.2)(l10)
        #y
        l11 = Add()([l7,l10])
        l11 = LeakyReLU(alpha = 0.2)(l11)

        l12=Conv2D(filters=1,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(l11)
        d = Flatten()(l12)
        d = Dense(64, activation='relu', kernel_initializer='he_uniform')(d)
        out = Dense(2, activation='softmax', kernel_initializer='he_uniform')(d)


        model = Model(inputs = inp, outputs = out)
        opt   = Adam(lr=0.0002, beta_1=0.5)
        model.compile(loss='binary_crossentropy', optimizer=opt)
        return model


    def _define_gan(self, gen_loss_in, gen_loss_wghts):
      g_model  = self.g_model
      d_model1 = self.d_model1
      d_model2 = self.d_model2
      image_shape = self.image_shape_gen

      for layer in d_model1.layers:
                layer.trainable = False
      d_model1.trainable = False
      for layer in d_model2.layers:
                layer.trainable = False
      d_model2.trainable = False

      # define the source image
      in_src = Input(shape=image_shape,  name="GAN_Input")


      gen_out = g_model(in_src)
      # dis_out = d_model([in_src, gen_out])
      dis_out1 = d_model1(gen_out)
      dis_out2 = d_model2(gen_out)

      # src image as input, generated image and classification output
      length = len(gen_loss_in)
      gen_set = [gen_out for x in range(length)]

      in_out = [dis_out1, dis_out2, gen_out, gen_out]
      #in_out = [dis_out1, dis_out2] + gen_set
      model  = Model(in_src, in_out, name="BCDUnetMultiScale_GAN")

      # Loss Adjustment
      dis_loss      = ['binary_crossentropy', 'binary_crossentropy']
      dis_loss_wght = [.5, 1]

      loss_in    = dis_loss      + gen_loss_in
      loss_wghts = dis_loss_wght + gen_loss_wghts

      if len(loss_in) != len(loss_wghts):
        print('Error!!! Mismatch in count of  loss and weight.')
      else:
        print('No of Generator losses: ' + str(len(gen_loss_wghts)))
        print('Loss Functions:', gen_loss_in)
        print('Loss Weights:  ', gen_loss_wghts)

      opt = Adam(lr=0.0002, beta_1=0.5)
      model.compile(loss=loss_in, loss_weights=loss_wghts, optimizer=opt)
                    
      return model
 