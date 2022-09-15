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
model.load_weights("path to the trained model weights with (.h5) extension")
print("Loaded model from disk")
#Display the model's architecture
model.summary()


if args["test_mode"] == "multiple":
  predict_multiple(
  model, 
  checkpoints_path="path to the trained model checkpoints", 
  inp_dir="path to the input fire images directory", 
  out_dir="path to the output segmented fire images directory")

elif args["test_mode"] == "video":
  predict_video(
  model,
  checkpoints_path="path to the trained model checkpoints", 
  inp="path to the input test video", 
  output="path to the output fire segmented video")
else:
  predict(
    model,
    checkpoints_path="path to the trained model checkpoints", 
    inp="path to the input fir image",
    out_fname="path to the output fire segmented image")


