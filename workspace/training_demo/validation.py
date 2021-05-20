import glob
import xml.etree.ElementTree as ET
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']= '-1'    # Don't use GPU
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from PIL import Image
import operator
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
matplotlib.use('TkAgg')

# load list of images
def load_images(dir_path):
    image_paths = []
    
    for filename in os.listdir(dir_path):
        image_path = dir_path + filename
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))

    return image_paths

# load model
def load_model(model_name):
    base_path = 'exported-models/'
    model_dir = base_path + model_name
    return str(model_dir)

# load label map
def load_label_map(filename):
    base_path = 'annotations/' # path to label map file
    label_dir = base_path + filename
    return str(label_dir)

def read_label_map(label_map_path):
    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None

    return items

def load_image_into_numpy_array(path):
    return np.array(Image.open(path))

'''
Hàm đếm số lượng nhãn đã gán của của 1 đối tượng
Input:
    dir_path: đường dẫn đến thư mục
    label_name: tên nhãn, ví dụ: bun_thit_nuong
Output:
    count: số lượng nhãn
'''
def count_nb_labels(dir_path, label_name):
    count = 0 # biến đếm số lượng nhãn

    for xml_file in glob.glob(dir_path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()

        for member in root.findall('object'):
            value = (member[0].text)
            if(value == label_name): # nếu value bằng với tên nhãn truyền vào
                count = count + 1 # thì tăng biến đếm lên 1
    
    # trả về số lượng nhãn
    return count

def get_nb_objects(base_path, path_to_labels):
    objects = {}
    names = read_label_map(path_to_labels)

    # duyệt 8 lớp
    for i in range(1,9):
        dir_name = names[i] + "_xml"
        # print(dir_name)
        path = base_path + dir_name
        # print(path)
        nb_objects = count_nb_labels(path, names[i])
        # print(names[i], ":", nb_objects)
        objects.update({names[i] : nb_objects})

    return objects

def cal_precisions_recalls(objects):
    precisions = {}
    recalls = {}

    sum_recalls = 0.0
    sum_precisions = 0.0

    for key, values in objects.items():
        # số lượng đối tượng nhận dạng đúng / số lượng đối tượng được nhận dạng (số lượng box)
        p = values[2]/values[1]
        # số lượng đối tượng nhận dạng đúng / số lượng đối tượng thực tế 
        r = values[2]/values[0]

        sum_precisions = sum_precisions + p
        sum_recalls = sum_recalls + r 

        precisions.update({key : p})
        recalls.update({key : r})

    avg_precisions = sum_precisions/len(objects)
    avg_recalls = sum_recalls/len(objects)

    return precisions, avg_precisions, recalls, avg_recalls

def cal_F1_score(precisions, recalls):
    F1_scores = {}
    sum_F1_scores = 0.0

    for key in precisions.keys():
        F1 = (2 * (precisions[key] * recalls[key])) / (precisions[key] + recalls[key])
        
        sum_F1_scores = sum_F1_scores + F1
        
        F1_scores.update({key : F1})

    avg_F1_scores = sum_F1_scores/len(precisions)

    return F1_scores, avg_F1_scores

def main():
    BASE_PATH = 'images/validation/'
    MODEL_NAME = 'my_efficientdet_model'
    PATH_TO_MODEL_DIR = load_model(MODEL_NAME)
    PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    LABEL_FILENAME = 'label_map.pbtxt'
    PATH_TO_LABELS = load_label_map(LABEL_FILENAME)

    names = read_label_map(PATH_TO_LABELS)

    # số lượng thực tế
    # objects = get_nb_objects(BASE_PATH, PATH_TO_LABELS)
    objects_new = {}
    
    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    # duyệt qua 8 lớp
    for i in range(1, 8):
        PATH_TO_IMAGE_DIR = BASE_PATH + names[i] + "/"
        # PATH_TO_IMAGE_DIR = BASE_PATH + "banh_khot/"

        TEST_IMAGES_PATH = load_images(PATH_TO_IMAGE_DIR)

        category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

        
        # đếm số lượng đối tượng nhận dạng đúng
        # vd: trong 200 ảnh bánh xèo, số lượng box là 250 nhưng trong 250 box chỉ có 200 đối tượng thực sự là bánh xèo
        count_nb_objects = 0
        # đếm số lượng box 
        count_nb_boxes = 0

        for image_path in TEST_IMAGES_PATH:
            
            
            print(image_path)

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
                    agnostic_mode=False
            )

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

                    # print(class_name)
                    # if class_name == names[i]:
                    #     count_nb_objects = count_nb_objects + 1
            
            objects_new.update({names[i] : [count_nb_boxes]})
            # plt.figure()
            # plt.imshow(image_np_with_detections)
        
        # plt.show()
        # print("LỚP", names[i])
        # print("Số lượng thực tế (gãn nhãn tay):", objects_new[names[i]][0])
        # print("Số lượng đối tượng nhận dạng (số lượng bounding box):", objects_new[names[i]][1])
        # print("Số lượng đối tượng nhận dạng đúng:", objects_new[names[i]][2])
        # print("=========================================================================\n")
   
    print(objects_new)
    '''p, avg_p, r, avg_r = cal_precisions_recalls(objects_new)
    print("Precision:")
    for key, value in p.items():
        print(key, ":", value)

    print("=========================================================================\n")

    print("Recall:")
    for key, value in r.items():
        print(key, ":", value)

    print("=========================================================================\n")

    F1, avg_F1 = cal_F1_score(p, r)
    print("F1 scores:")
    for key, value in F1.items():
        print(key, ":", value)
    
    print("\nMAP")
    print("Precision:", avg_p)
    print("Recall:", avg_r)
    print("F1 score:", avg_F1)'''

if __name__=="__main__":
    main()