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
img_path = './workspace/testset1'
def main(_):
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]
    img_list = os.listdir(img_path)
    print(img_list)

    print(img_list[1])

    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        logits = sess.graph.get_tensor_by_name('final_result:0')
        print('=====================      예측결과      ======================')
        for i in img_list:
            image = tf.gfile.FastGFile(img_path+'/'+ i, 'rb').read()
            prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})
            print(i)
            print('---------------------------------------------------------------')
            it_is=''
            temp =0
            for j in range(len(labels)):
                name = labels[j]
                score = prediction[0][j]
                if(score>temp):
                    temp=score
                    it_is=name
                print('%s (%.2f%%)' % (name, score * 100))
            print('---------------------------------------------------------------')
            if it_is in i:
                print(">>result:correct")
            else:
                print(">>result:wrong")
            # print('It is %s (%.2f%%)' % (it_is, temp * 100))
            print('=============================================================== \n')


    # if FLAGS.show_image:
    #     img = mpimg.imread('./workspace/flower_photos/testing34.jpg')
    #     plt.imshow(img)
    #     plt.show()


if __name__ == "__main__":
    tf.app.run()










