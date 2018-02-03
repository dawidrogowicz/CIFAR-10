from tflearn import input_data, conv_2d, max_pool_2d, fully_connected, regression, dropout, DNN, Adam
adam = Adam(beta2=.998)


def model(size=32, color_channels=3):
    convnet = input_data((None, size, size, color_channels))
    convnet = conv_2d(convnet, 12, 5, activation='relu')
    convnet = conv_2d(convnet, 12, 5, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = conv_2d(convnet, 18, 3, activation='relu')
    convnet = conv_2d(convnet, 16, 3, activation='relu')
    convnet = max_pool_2d(convnet, 2)
    convnet = fully_connected(convnet, 1024, 'relu')
    convnet = fully_connected(convnet, 256, 'relu')
    convnet = dropout(convnet, .95)
    convnet = fully_connected(convnet, 10, 'softmax')
    convnet = regression(convnet, optimizer=adam)

    return DNN(convnet, tensorboard_verbose=1, tensorboard_dir='tflearn_logs')
