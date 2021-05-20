# Hệ thống nhận dạng các món ăn truyền thống miền Nam

# Installation
https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

# Download the dataset
- Download the dataset at https://drive.google.com/drive/folders/1yjAQo1pYQUl8mKn1GTbYPE1NAtWpJsVo?usp=sharing. 
- Extracting and puting images in workspace/training_demo directory.
- Extracting and puting record files in workspace/training_demo/annotations directory.

# Running
```
cd workspace/training_demo
python run.py
```

# Training an Object Detection model using Tensorflow Object Detection API

## Preparing dataset
Partition the dataset and create Tensorflow records: see more details in script/preprocessing directory

## Checking the dataset before starting the training phase
```
cd workspace/training_demo
python counting_labels.py
```

## Download pre-trained model
- Creating new directory named "pre-trained-models" in the workspace/training_demo directory
- Downloading the EfficientDet D0 512x512 at https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
- Putting the downloaded file in this workspace/training_demo/pre-trained-models, then extract it.
- Opening your terminal to run file plot_object_detection_saved_model.py to testing the downloaded model
```
# change directory
cd workspace/training_demo

# activate conda environment
conda activate tensorflow

# run python code
python plot_object_detection_saved_model.py
```

## Training custom Object detection

1. Training the Model
```
# change directory
cd workspace/training_demo

# activate conda environment
conda activate tensorflow

# run python code
python model_main_tf2.py --model_dir=models/my_efficientdet --pipeline_config_path=models/my_efficientdet/pipeline.config
```

2. Exporting a Trained model
- Creating new directory named "exported-models/my_models" in the workspace/training_demo directory
- Opening and running the following commands in your terminal
```
# change directory
cd workspace/training_demo

# activate conda environment
conda activate tensorflow

# run python code
python exporter_main_v2.py --input_type image_tensor --pipeline_config_path models/my_efficientdet/pipeline.config --trained_checkpoint_dir models/my_efficientdet/ --output_directory exported-models/my_efficientdet_model/
```

3. Testing the exported model
```
# change directory
cd workspace/training_demo

# activate conda environment
conda activate tensorflow

# run python code
python test_object_detection.py
```

# Training an Object Detection model using Google Colab
Using computer_vision.ipynb
