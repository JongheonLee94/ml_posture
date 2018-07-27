#you have to classfy directory = image name in testset
import tensorflow as tf
# import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys


from tensorflow.python.platform import gfile
import os.path
import re
import hashlib
from tensorflow.python.util import compat

MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

tf.app.flags.DEFINE_string("output_graph",
                           "./workspace/posture_graph.pb",
                           "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels",
                           "./workspace/posture_labels.txt",
                           "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean("show_image",
                            True,
                            "이미지 추론 후 이미지를 보여줍니다.")

FLAGS = tf.app.flags.FLAGS
images = [ ]
img_path = './workspace/testset2'
list_len= 0
def main(_):
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]
    img_list = os.listdir(img_path)
    print(img_list)
    list_len=len(img_list)
    print(list_len)
    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        correct_counter = 0
        wrong_counter = 0
        list_len = len(img_list)

        logits = sess.graph.get_tensor_by_name('final_result:0')
        print('=====================      예측결과      ======================')
        for i in img_list:
            image = tf.gfile.FastGFile(img_path + '/' + i, 'rb').read()
            prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})
            print('===============================================================')
            print(i)
            print('---------------------------------------------------------------')
            it_is = ''
            temp = 0
            for j in range(len(labels)):
                name = labels[j]
                score = prediction[0][j]
                if (score > temp):
                    temp = score
                    it_is = name
                print('%s (%.2f%%)' % (name, score * 100))
            print('---------------------------------------------------------------')
            if it_is in i:
                print(">>result:correct")
                correct_counter = correct_counter + 1
            else:
                print(">>result:wrong")
                wrong_counter = wrong_counter + 1
            # print('It is %s (%.2f%%)' % (it_is, temp * 100))
            print('=============================================================== \n')

        print('Data:%d (Correct:%d, Wrong: %d)     Accuracy:%.2f%%' % (
        list_len, correct_counter, wrong_counter, correct_counter / list_len * 100))

    # if FLAGS.show_image:
    #     img = mpimg.imread('./workspace/flower_photos/testing34.jpg')
    #     plt.imshow(img)
    #     plt.show()


if __name__ == "__main__":
    tf.app.run()









