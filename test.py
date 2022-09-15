from keras_segmentation.models.unet import shufflnet_unet

from keras_segmentation.data_utils.visualize_dataset import *
from keras_segmentation.predict import predict_multiple, predict_video, predict
from keras_segmentation.predict import model_from_checkpoint_path

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-tm", "--test_mode", required=True,
	help="defines the test mode: single image|multiple images|video")

args = vars(ap.parse_args())

model = shufflnet_unet(n_classes=2,  input_height=416, input_width=608)
model.load_weights("F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/trained_weights/shufflenet_unet11_.h5")
print("Loaded model from disk")
#Display the model's architecture
model.summary()


if args["test_mode"] == "multiple":
  predict_multiple(
  model, 
  checkpoints_path="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/trained_weights/shufflenet_unet11_", 
  inp_dir="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/new_test_images/", 
  out_dir="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/shufflenet_unet_results/shufflenet_unet12/")

elif args["test_mode"] == "video":
  predict_video(
  model,
  checkpoints_path="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/trained_weights/shufflenet_unet11_", 
  inp="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/new_test_images/fire_video.mp4", 
  output="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/shufflenet_unet_results/shufflenet_unet12/fire_seg_video.mp4")
else:
  predict(
    model,
    checkpoints_path="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/unet/shufflenet_unet11/shufflenet_unet11_", 
    inp="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/new_test_images/fire007.png",
    out_fname="F:/Hayat Data/Hayat master's PC backup/drive volume G/image-segmentation-keras-master/fire_segmentation/shufflenet_unet_results/shufflenet_unet12/fire007.png")


