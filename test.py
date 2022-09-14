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
from keras_segmentation.predict import predict_multiple, predict_video, predict
from keras_segmentation.predict import model_from_checkpoint_path



model = shufflnet_unet(n_classes=2,  input_height=416, input_width=608)
model.load_weights("F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/unet/shufflenet_unet11/shufflenet_unet11_.h5")
print("Loaded model from disk")
#model=model_from_checkpoint_path("F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/unet/shufflenet_unet11/shufflenet_unet11_")
#Display the model's architecture
model.summary()





'''
out = model.predict_multiple( 
  checkpoints_path="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/unet/shufflenet_unet11/shufflenet_unet11_", 
  inp_dir="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/new_test_images/", 
  out_dir="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/shufflenet_unet_results/shufflenet_unet11/" 
)
'''




predict_video(
  model,
  checkpoints_path="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/unet/shufflenet_unet11/shufflenet_unet11_", 
  inp="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/new_test_images/fire_video.mp4", 
  output="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/shufflenet_unet_results/shufflenet_unet11/fire_seg_video.mp4" 
)


'''
out = model.predict_segmentation(
    checkpoints_path="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/unet/shufflenet_unet11/shufflenet_unet11_", 
    inp="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/new_test_images/fire007.png",
    out_fname="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/shufflenet_unet_results/shufflenet_unet11//shufflenet_unet11_fire007.png"
)
'''


