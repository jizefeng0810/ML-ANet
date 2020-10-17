import xml.etree.ElementTree as ET
import pickle
import os
import numpy as np
from os import listdir, getcwd
from os.path import join

Absolute_path = 'G:/Now_Project/prepare_data/'

sets=[('2007', 'source_train'),('2007', 'target_train'),('2007', 'target_test')]
# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        #obj_struct['pose'] = obj.find('pose').text
        #obj_struct['truncated'] = int(obj.find('truncated').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(float(bbox.find('xmin').text)),
                              int(float(bbox.find('ymin').text)),
                              int(float(bbox.find('xmax').text)),
                              int(float(bbox.find('ymax').text))]
        objects.append(obj_struct)
    return objects

cls = np.zeros((8))
cls_num = []
if __name__ == '__main__':
    for year, image_set in sets:
        if not os.path.exists(Absolute_path + 'VOCdevkit%s/VOC%s/labels/' % (year, year)):  # 创建labels文件夹
            os.makedirs(Absolute_path + 'VOCdevkit%s/VOC%s/labels/' % (year, year))
        image_ids = open(Absolute_path + 'VOCdevkit%s/VOC%s/ImageSets/Main/%s.txt' % (year, year, image_set)).read().strip().split()  # 打开txt文件读取图片文件名
        out_file = open(Absolute_path + 'VOCdevkit%s/VOC%s/labels/%s_3.txt' % (year, year, image_set), 'w')
        object_num = 1  # 有标签类
        no_object_num = 0   # 无标签类
        for image_id in image_ids:
            xml_file = open(Absolute_path + 'VOCdevkit%s/VOC%s/Annotations/%s.xml' % (year, year, image_id))
            results = parse_rec(xml_file)
            if len(results) == 0:
                no_object_num+=1
                continue
            out_file.write('%s.jpg'%(image_id))
            for result in results:
                class_name = result['name']
                bbox = result['bbox']
                class_encoder = classes.index(class_name)
                if class_encoder == 0 or class_encoder == 1:    # 统计类别出现数量
                    cls[0] = 1
                elif class_encoder == 2 or class_encoder == 3 or class_encoder == 4:
                    cls[1] = 1
                elif class_encoder == 6 or class_encoder == 7:
                    cls[2] = 1
                out_file.write(' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3]) + ' ' + str(class_encoder))
            out_file.write('\n')
            object_num += 1
            cls_num.append(cls)
            cls = [0,0,0,0, 0,0,0,0]
        aa = np.sum(cls_num, axis=0)
        print(aa)
    print('data init')



