
'''
Creates the data layer. It creates a data_layer.protxt that it is not used. Its content is the data layer calling code
and should be copied into the train.prototxt and val.protxt
TODO: Check how Caffe interacts with the class of the created layer
'''


from create_net import build_net

split_train = 'trainCitiesInstagram'
split_val = 'valCitiesInstagram'
num_labels = 100
batch_size = 1 #AlexNet 100, VGG 40
resize_w = 300
resize_h = 300
crop_w = 224 #227 AlexNet, 224 VGG16
crop_h = 224
crop_margin = 10 #The crop won't include the margin in pixels
mirror = True #Mirror images with 50% prob
rotate = 0 #15,8 #Always rotate with angle between -a and a
HSV_prob = 0 #0.3,0.15 #Jitter saturation and vale of the image with this prob
HSV_jitter = 0 #0.1,0.05 #Saturation and value will be multiplied by 2 different random values between 1 +/- jitter


#JUSTO TO CREATE DATA LAYER; NET ARQ IS HARDCODED
build_net(split_train, num_labels, batch_size, resize_w, resize_h, crop_w, crop_h, crop_margin, mirror, rotate, HSV_prob, HSV_jitter, train=True)