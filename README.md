# Fire Segmentation
Efficient Fire Segmentation for Internet-of_Things-Assisted Intelligent Transportation Systems 

![](readme_images/framework.png)

## Introduction
This repo contains the implementation of our proposed fire segmentation method, the prerequisite libraries, and link of our newly created dataset we have used in our experiments. For testing purpose we also provide a trained model weights and test fire images.

---
This code is written with Anaconda python 3.7. Install the Anaconda python 3.7 and clone the repo with the following command:
```
git clone https://github.com/hayatkhan8660-maker/Fire_Seg_Dataset.git
cd Fire_Seg_Dataset
```

----
### Dependencies
This code requires the following libraries, install these libraries before testing code. 
- keras_segmentation
- numpy == 1.21.6
- h5py == 2.10.0
- opencv-python == 4.6.0
- tqdm == 4.64.0
- tensorflow == 2.9.1
- keras == 2.9.0

run ```pip install -r requirements.txt``` to install all the dependencies. 

### Our Fire Segmentation Dataset and its Training Setup
Our fire segmentation dataset consist of fire images and their corresponding annotated fire masks. For training, we divided the dataset into two subsets i.e., train and test sets. Where each set contains fire images and their corresponding annotated fire masks. [Dataset Link](https://drive.google.com/drive/folders/1Xfq7zLwIwJ4vPx50G-k7j2-ofh1bj3fx?usp=sharing)

The structure of the dataset directory should be as follows:

```
Fire Dataset
├── images_prepped_train
│   ├── img(1).jpg
│   ├── img(2).jpg
│   ├── img(3).jpg
│   ├── .....
├── annotations_prepped_train
│   ├── img(1).png
│   ├── img(2).png
│   ├── img(3).png
│   ├── .....
├── images_prepped_test
│   ├── img(1).jpg
│   ├── img(2).jpg
│   ├── img(3).jpg
│   ├── .....
├── annotations_prepped_test
│   ├── img(1).png
│   ├── img(2).png
│   ├── img(3).png
│   ├── .....

```

## Training
Run the following command to train the proposed fire segmentation model on our fire segmentation dataset.
```
python train.py --train_images "path to input training fire images" 
--train_annotations "path to input training annotations fire masks"
--validation_images "path to input validation fire images"
--validation_annotations "path to input validation annotations fire masks"
--checkpoints_path "path to output training checkpoints"
--trained_weights "path to output training weights"
--epochs number_of_epochs
```

## Test
Run the following command to test the proposed trained model for fire segmentation task on the images from test set.
```
python test.py --test_mode "single|multiple|video"
```
User can decide the test_mode by providing appropriate argument.

- single: it run the test.py for single image fire segmentation.
- multiple: it run the test.py for multiple images fire segmentation based on the provided test images directory.
- video: it run the test.py for fire segmentation in video.

## Fire Extraction
Run the following command to extract fire specific regions from the input fire image using model predicted segmentation mask.

```
python fire_extraction.py --input_image "input fire image" 
--segmented_image "corresponding model segmented image"
--fire_specific_image "fire sepecific pixels image"

```

## Obtained Visual Results from Conducted Experiments 
Following are the visual results obtained by our proposed fire segmentation method from different experimental setups.

![](readme_images/paper_figure2.png)

![](readme_images/paper_figure3.png)

![](readme_images/paper_figure4.png)

![](readme_images/paper_figure5.png)

![](readme_images/paper_figure6.png)

## Obtained Results on Video 

![](test_images/test_video.gif)   ![](Visual_output_results/ouput_segmented_video.gif)

<img src="test_images/test_video.gif" width="425"/> <img src="Visual_output_results/ouput_segmented_video.gif" width="425"/> 
