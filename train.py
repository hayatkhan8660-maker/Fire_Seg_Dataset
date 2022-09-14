# segnet evaluation

#from keras_segmentation.models.segnet import segnet
#from keras_segmentation.models.segnet import vgg_segnet
#from keras_segmentation.models.segnet import resnet50_segnet
#from keras_segmentation.models.segnet import mobilenet_segnet

# unet evaluation
from keras_segmentation.models.unet import shufflnet_unet
#from keras_segmentation.models.unet import unet
#from keras_segmentation.models.unet import vgg_unet
#from keras_segmentation.models.unet import resnet50_unet
#from keras_segmentation.models.unet import mobilenet_unet
#from keras_segmentation.models.unet import unet_mini
#from keras_segmentation.models.unet import _unet

#from keras_segmentation.models.pspnet import pspnet
#from keras_segmentation.models.pspnet import vgg_pspnet
#from keras_segmentation.models.pspnet import resnet50_pspnet
#from keras_segmentation.models.pspnet import pspnet_50
#from keras_segmentation.models.pspnet import pspnet_101
#from keras_segmentation.models.pspnet import squeeze_pspnet
#from keras_segmentation.models.pspnet import mobile_pspnet

#from keras_segmentation.models.fcn import fcn_8
#from keras_segmentation.models.fcn import fcn_8_vgg
#from keras_segmentation.models.fcn import fcn_8_mobilenet
from keras_segmentation.data_utils.visualize_dataset import *
from keras_segmentation.predict import predict_multiple
from keras_segmentation.predict import model_from_checkpoint_path


#train_images =  "F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/test/example_dataset/images_prepped_train/"
#train_annotations = "F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/test/example_dataset/annotations_prepped_train/"

#visualize_segmentation_dataset(train_images,train_annotations,2)


model = shufflnet_unet(n_classes=2,  input_height=416, input_width=608)
#model.load_weights('H:/image-segmentation-keras-master/traind_weights/vgg_unet_1.h5')
#print("Loaded model from disk")s
#model=model_from_checkpoint_path("H:/image-segmentation-keras-master/traind_weights/vgg_unet_1")
#Display the model's architecture
model.summary()

model.train(
    train_images =  "F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/test/example_dataset/images_prepped_train/",
    train_annotations = "F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/test/example_dataset/annotations_prepped_train/",
    val_images = "F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/test/example_dataset/images_prepped_test",
    val_annotations = "F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/test/example_dataset/annotations_prepped_test",
    optimizer_name= 'SGD',
    checkpoints_path = "F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/unet/shufflenet_unet11/shufflenet_unet11_", 
    epochs=50)

model.save("F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/unet/shufflenet_unet11/shufflenet_unet11_.h5") 



'''
predict_multiple( 
  checkpoints_path="H:/image-segmentation-keras-master/traind_weights/vgg_unet_1", 
  inp_dir="H:/image-segmentation-keras-master/test/example_dataset/images_prepped_test/" , 
  out_dir="H:/image-segmentation-keras-master/Segmentation_results/" 
)
'''
'''
out = model.predict_segmentation(
    checkpoints_path="H:/image-segmentation-keras-master/traind_weights/vgg_unet_1", 
    inp="H:/image-segmentation-keras-master/new_test_images/fire060.png",
    out_fname="H:/image-segmentation-keras-master/new_results/mobile_fire007.png"
)
'''

'''
import matplotlib.pyplot as plt
plt.imshow(out)
'''
# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/test/example_dataset/images_prepped_test", 
annotations_dir="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/test/example_dataset/annotations_prepped_test" ))
