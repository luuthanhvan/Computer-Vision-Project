import glob
import xml.etree.ElementTree as ET
import numpy as np

'''
Hàm lấy danh sách tên của các đối tượng đã gán nhãn
Input:
    dir_path: đường dẫn đến thư mục
Output:
    object_list: mảng lưu trữ toàn bộ tên nhãn của các đối tượng đã gán nhãn
'''
def get_list_of_objects(dir_path):
    list_of_objects = [] # mảng lưu trữ tên nhãn của các đối tượng
    count_xml_file = 0

    # duyệt qua toàn bộ file xml trong dir_path
    for xml_file in glob.glob(dir_path + '/*.xml'):
        tree = ET.parse(xml_file) # tạo element tree object 
        root = tree.getroot() # lấy root element 
        
        count_xml_file = count_xml_file + 1

        # duyệt qua các member mà nó có tên là object trong root element
        for member in root.findall('object'):
            value = (member[0].text)
            # thêm tên nhãn vào trong mảng list_of_objects
            # if value == "banh_khoy":
            #     print(xml_file)
            list_of_objects.append(value)

            # if(value == ''):
            #     for member_2 in root.findall('filename'):
            #         print(member_2.text)

    # Bung dòng 27 ra để xem toàn bộ các giá trị lưu trong mảng object_list
    # print(object_list)
    
    # print("Number of xml files:", count_xml_file)

    # trả về mảng các tên nhãn
    return list_of_objects

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

def get_list_of_nb_labels():
    PATH_TO_IMAGE_DIR = 'images/validation'
    
    objects = {}

    list_of_objects_unique = np.unique(get_list_of_objects(PATH_TO_IMAGE_DIR))

    for i in range(len(list_of_objects_unique)):
        label_name = list_of_objects_unique[i]
        nb_labels = count_nb_labels(PATH_TO_IMAGE_DIR, label_name)
        objects.update({label_name: nb_labels}) 

    return objects


def main():
    # objects = get_list_of_nb_labels()
    # print(objects)

    # đường dẫn đến thư mục train
    PATH_TO_TRAIN_DIR = 'images/dataset/train'
    # đường dẫn đến thư mục test
    PATH_TO_TEST_DIR = 'images/dataset/test'
    # đường dẫn đến thư mục validation
    PATH_TO_VAL_DIR = 'images/validation'

    # PATH_TO_IMAGE_DIR = 'images/xml_files_validation'
    
    # list_of_objects_unique = np.unique(get_list_of_objects(PATH_TO_IMAGE_DIR))
    # print('List of objects:', list_of_objects_unique)
    # print('Number of objects:', len(list_of_objects_unique))

    # print('IN IMAGES DIRECTORY:')
    # for i in range(len(list_of_objects_unique)):
    #     label_name = list_of_objects_unique[i]
    #     nb_labels = count_nb_labels(PATH_TO_IMAGE_DIR, label_name)
    #     print('Number of label ' + label_name + ': ' + str(nb_labels))

    # xét trong thư mục train
    print('IN TRAIN DIRECTORY:')
    # hàm unique là loại bỏ các gía trị trùng trong mảng và kết quả của hàm unique là 1 mảng
    list_of_objects_unique = np.unique(get_list_of_objects(PATH_TO_TRAIN_DIR))
    print('List of objects:', list_of_objects_unique)
    print('Number of objects:', len(list_of_objects_unique))
    # mảng list_of_objects_unique có dạng ['banh_khot' 'banh_pia' 'banh_tet' 'banh_xeo' 'bun_mam' 'bun_thit_nuong' 'com_tam' 'goi_cuon']
    # do đó mình sẽ lấy từng phần tử banh_khot, banh_pia... là các label_name để truyền vào hàm count_nb_labels()
    for i in range(len(list_of_objects_unique)):
        label_name = list_of_objects_unique[i]
        nb_labels = count_nb_labels(PATH_TO_TRAIN_DIR, label_name)
        print('Number of label ' + label_name + ': ' + str(nb_labels))

    # xét trong thư mục test
    print("\nIN TEST DIRECTORY:")
    list_of_objects_unique = np.unique(get_list_of_objects(PATH_TO_TEST_DIR))
    print('List of objects:', list_of_objects_unique)
    print('Number of objects:', len(list_of_objects_unique))
    
    for i in range(len(list_of_objects_unique)):
        label_name = list_of_objects_unique[i]
        nb_labels = count_nb_labels(PATH_TO_TEST_DIR, label_name)
        print('Number of label ' + label_name + ': ' + str(nb_labels))

    # xét trong thư mục validation
    print("\nIN VALIDATION DIRECTORY:")
    list_of_objects_unique = np.unique(get_list_of_objects(PATH_TO_VAL_DIR))
    print('List of objects:', list_of_objects_unique)
    print('Number of objects:', len(list_of_objects_unique))
    
    for i in range(len(list_of_objects_unique)):
        label_name = list_of_objects_unique[i]
        nb_labels = count_nb_labels(PATH_TO_VAL_DIR, label_name)
        print('Number of label ' + label_name + ': ' + str(nb_labels))

if __name__=="__main__":
    main()