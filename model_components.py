def get_generator_model(z, reuse=False, training_mode=False):
    
    print("\nGenerator:\n")
    print("Input shape of z is {}".format(z.shape))
    
    dcgan_kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.002)

    with tf.variable_scope('generator', reuse=reuse):
        
        # inp.shape (100)
        # out.shape (8*8*1024)
        name='layer_01'
        with tf.variable_scope(name): 
            z = tf.layers.dense(inputs=z, units=8*8*1024, kernel_initializer=dcgan_kernel_initializer)
            # reshape to (batch_size,8,8,1024)
            z = tf.reshape(z, (-1,8,8,1024))
            z = tf.layers.batch_normalization(inputs=z)
            z = tf.nn.leaky_relu(features=z, alpha=0.2)
            print("Output shape of {} is {}".format(name, z.shape))
            
        
        # in.shape (8,8,1024)
        # out.shape (16,16,512)
        name='layer_02'
        with tf.variable_scope(name):
            z = tf.layers.conv2d_transpose(inputs=z,
                                           filters=512,
                                           kernel_size=(5,5),
                                           strides=(2,2),
                                           padding='SAME',
                                           kernel_initializer = dcgan_kernel_initializer
                                                )
            z = tf.layers.batch_normalization(inputs=z, training=training_mode)
            z = tf.nn.leaky_relu(features=z, alpha=0.2)
            print("Output shape of {} is {}".format(name, z.shape))
                                 
        
        # in.shape (16,16,512)
        # out.shape (32,32,256)
        name='layer_03'
        with tf.variable_scope(name):
            z = tf.layers.conv2d_transpose(inputs=z,
                                           filters=256,
                                           kernel_size=(5,5),
                                           strides=(2,2),
                                           padding='SAME',
                                           kernel_initializer = dcgan_kernel_initializer
                                                )
            z = tf.layers.batch_normalization(inputs=z, training=training_mode)
            z = tf.nn.leaky_relu(features=z, alpha=0.2)
            print("Output shape of {} is {}".format(name, z.shape))
                                 
            
        # in.shape (32,32,256)
        # out.shape (64,64,128)
        name='layer_04'
        with tf.variable_scope(name):
            z = tf.layers.conv2d_transpose(inputs=z,
                                           filters=128,
                                           kernel_size=(5,5),
                                           strides=(2,2),
                                           padding='SAME',
                                           kernel_initializer = dcgan_kernel_initializer
                                                )
            z = tf.layers.batch_normalization(inputs=z, training=training_mode)
            z = tf.nn.leaky_relu(features=z, alpha=0.2)
            print("Output shape of {} is {}".format(name, z.shape))
                                 
                                 
        # in.shape (64,64,128)
        # out.shape (128,128,64)
        name='layer_05'
        with tf.variable_scope(name):
            z = tf.layers.conv2d_transpose(inputs=z,
                                           filters=64,
                                           kernel_size=(5,5),
                                           strides=(2,2),
                                           padding='SAME',
                                           kernel_initializer = dcgan_kernel_initializer
                                                )
            z = tf.layers.batch_normalization(inputs=z, training=training_mode)
            z = tf.nn.leaky_relu(features=z, alpha=0.2)
            print("Output shape of {} is {}".format(name, z.shape))
                                 
                        
        # in.shape (128,128,64)
        # out.shape (256,256,3)
        name='layer_06'
        with tf.variable_scope(name):
            z = tf.layers.conv2d_transpose(inputs=z,
                                           filters=3,
                                           kernel_size=(5,5),
                                           strides=(2,2),
                                           padding='SAME',
                                           kernel_initializer = dcgan_kernel_initializer
                                                )
            z = tf.tanh(z)
            print("Output shape of {} is {}".format(name, z.shape))
        
        
        return z


def get_discriminator_model(x, z, reuse=False, training_mode=False, penultimate_layer_units=1024):
    
    '''
    For Real Images:
    Input: x, E(x) 
    Output: Probability of image being real
    
    For Generated Images:
    Input: G(z), z 
    Output: Probability of real image
    '''
    
    print("\nDiscriminator: \n")
    print("Input shape of x is {}".format(x.shape))
    
    dcgan_kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.002)

    with tf.variable_scope('discriminator', reuse=reuse):

        # inp.shape (256,256,3)
        # out.shape (128,128,64)
        name='x_layer_01'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=64, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer = dcgan_kernel_initializer
                             )
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))
            

        # inp.shape (128,128,64)
        # out.shape (64,64,128)
        name='x_layer_02'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=128, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer = dcgan_kernel_initializer
                             )
            x = tf.layers.batch_normalization(inputs=x, training=training_mode)
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))
            

        # inp.shape (64,64,128)
        # out.shape (32,32,256)
        name='x_layer_03'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=256, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer = dcgan_kernel_initializer
                             )
            x = tf.layers.batch_normalization(inputs=x, training=training_mode)
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))


        # inp.shape (32,32,256)
        # out.shape (16,16,512)
        name='x_layer_04'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=512, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer = dcgan_kernel_initializer
                             )
            x = tf.layers.batch_normalization(inputs=x, training=training_mode)
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))


        # inp.shape (16,16,512)
        # out.shape (8,8,1024)
        name='x_layer_05'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=1024, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer=dcgan_kernel_initializer
                             )
            x = tf.layers.batch_normalization(inputs=x, training=training_mode)
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))

        x = tf.reshape(x, (-1, 8*8*1024))
        
        
        print("\nInput shape of z is {}".format(z.shape))
        
        # inp.shape (200)
        # out.shape (1024)
        name='z_layer_01'
        with tf.variable_scope(name):
            z = tf.layers.dense(inputs=z,
                                units=1024,
                                kernel_initializer=dcgan_kernel_initializer
                               )
            z = tf.layers.batch_normalization(inputs=z, training=training_mode)
            z = tf.nn.leaky_relu(features=z, alpha=0.2)
            z = tf.layers.dropout(inputs=z, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, z.shape))
            
        
    
        # z inp.shape (1024)
        # x inp.shape (8*8*1024)
        # concat[x,z] out.shape ()
        xz = tf.concat([x,z], axis=1)
        print("\nOutput shape of [x,z] concat is {}".format(xz.shape))
        
        
        # inp.shape (66560)
        # out.shape (penultimate_layer=1024)
        name='xz_layer_01'
        with tf.variable_scope(name):
            xz = tf.layers.dense(inputs=xz,
                                units=penultimate_layer_units,
                                kernel_initializer=dcgan_kernel_initializer
                               )
            xz = tf.layers.batch_normalization(inputs=xz, training=training_mode)
            xz = tf.nn.leaky_relu(features=xz, alpha=0.2)
            xz = tf.layers.dropout(inputs=xz, rate=0.5, training=training_mode)
        print("Output shape of {} is {}".format(name, xz.shape))
        
        penultimate_layer = xz
        
        
        # inp.shape (penultimate_layer=1024)
        # out.shape (1)
        name='xz_layer_02'
        with tf.variable_scope(name):
            xz = tf.layers.dense(inputs=xz,
                                units=1,
                                kernel_initializer=dcgan_kernel_initializer
                               )
            xz = tf.layers.batch_normalization(inputs=xz, training=training_mode)
            xz = tf.nn.leaky_relu(features=xz, alpha=0.2)
            xz = tf.layers.dropout(inputs=xz, rate=0.5, training=training_mode)
        print("Output shape of {} is {}".format(name, xz.shape))
        
        
        return xz, penultimate_layer


def get_encoder_model(x, latent_dimensions, reuse=False, training_mode=False):
    
    '''
    For Real Images:
    Input: x 
    Output: E(x)
    
    For Generated Images:
    Input: G(z) 
    Output: E(G(z))
    '''
    
    print("\nEncoder: \n")
    print("Input shape of x is {}".format(x.shape))
    
    dcgan_kernel_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.002)

    with tf.variable_scope('Encoder', reuse=reuse):
        
        # inp.shape (256,256,3)
        # out.shape (128,128,16)
        name='layer_01'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=16, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer = dcgan_kernel_initializer
                             )
            x = tf.layers.batch_normalization(inputs=x, training=training_mode)
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))
            
            
        # inp.shape (128,128,16)
        # out.shape (64,64,32)
        name='layer_02'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=32, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer = dcgan_kernel_initializer
                             )
            x = tf.layers.batch_normalization(inputs=x, training=training_mode)
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))
            
            
        # inp.shape (64,64,32)
        # out.shape (32,32,64)
        name='layer_03'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=64, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer = dcgan_kernel_initializer
                             )
            x = tf.layers.batch_normalization(inputs=x, training=training_mode)
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))
            
        
        # inp.shape (32,32,64)
        # out.shape (16,16,128)
        name='layer_04'
        with tf.variable_scope(name):     
            x = tf.layers.conv2d(inputs=x, 
                                 filters=128, 
                                 kernel_size=(5,5), 
                                 strides=(2,2), 
                                 padding='SAME', 
                                 kernel_initializer = dcgan_kernel_initializer
                             )
            x = tf.layers.batch_normalization(inputs=x, training=training_mode)
            x = tf.nn.leaky_relu(features=x, alpha=0.2)
            x = tf.layers.dropout(inputs=x, rate=0.5, training=training_mode)
            print("Output shape of {} is {}".format(name, x.shape))
            
        
        # inp.shape (16,16,128)
        # out.shape (latent dimesnions)
        name='layer_04'
        with tf.variable_scope(name):     
            x = tf.layers.flatten(x)
            x = tf.layers.dense(inputs=x, 
                                units=latent_dimensions,
                                kernel_initializer=dcgan_kernel_initializer
                               )
            print("Output shape of {} is {}".format(name, x.shape))
            
        return x


def test_components():

    
    # Test Generator
    placeholder = tf.placeholder(dtype=tf.float32, shape=(None,200))
    gen = get_generator_model(placeholder, tf.AUTO_REUSE)

    # Test Discriminator
    placeholder = tf.placeholder(dtype=tf.float32, shape=(None,256,256,3))
    placeholder2 = tf.placeholder(dtype=tf.float32, shape=(None,200))
    dis = get_discriminator_model(placeholder, placeholder2,tf.AUTO_REUSE)
    
    # Test Encoder
    placeholder = tf.placeholder(dtype=tf.float32, shape=(None,256,256,3))
    en = get_encoder_model(placeholder, latent_dimensions=200, reuse=tf.AUTO_REUSE)

    
    tf.reset_default_graph()
