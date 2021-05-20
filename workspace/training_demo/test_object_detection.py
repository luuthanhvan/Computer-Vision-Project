import os
os.environ['CUDA_VISIBLE_DEVICES']= '-1'    # Don't use GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import numpy as np
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# load list of images
def load_images():
    base_path = 'images/validation/test/' # path to image
    image_paths = []
    
    for filename in os.listdir(base_path):
        image_path = base_path + filename
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))

    print("test")
    print(image_paths)

    return image_paths

IMAGE_PATHS = load_images()

# load model
def load_model(model_name):
    base_path = 'exported-models/'
    model_dir = base_path + model_name
    return str(model_dir)

MODEL_NAME = 'my_efficientdet_model'
PATH_TO_MODEL_DIR = load_model(MODEL_NAME)

# load label map
def load_label_map(filename):
    base_path = 'annotations/' # path to label map file
    label_dir = base_path + filename
    return str(label_dir)

LABEL_FILENAME = 'label_map.pbtxt'
PATH_TO_LABELS = load_label_map(LABEL_FILENAME)
PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))

def main():
    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
    
    for image_path in IMAGE_PATHS:
        count_nb_boxes = 0

        print('Running inference for {}... '.format(image_path), end='')

        image_np = load_image_into_numpy_array(image_path)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

        # This is the way I'm getting my coordinates
        boxes = detections['detection_boxes']

        # get all boxes from an array
        max_boxes_to_draw = boxes.shape[0]
        
        # get scores to get a threshold
        scores = detections['detection_scores']
        
        # this is set as a default but feel free to adjust it to your needs
        min_score_thresh=.30
        
        # iterate over all objects found
        for j in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[j] > min_score_thresh:
                # boxes[j] is the box which will be drawn
                
                class_name = category_index[detections['detection_classes'][j]]['name']
                # print ("This box is gonna get used", boxes[j], detections['detection_classes'][j])
                count_nb_boxes = count_nb_boxes + 1

        print('\nNumber of boxes of {} is {}'.format(image_path, count_nb_boxes))

        plt.figure()
        plt.imshow(image_np_with_detections)
        print('Done')

    plt.show()

# calling main function
if __name__=='__main__':
    main()