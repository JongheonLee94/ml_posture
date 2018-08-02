# you have to classfy directory = image name in testset
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
images = []
dir_path = './workspace/'
testset = 'testset6'
img_path = dir_path + testset
list_len = 0

def result_of_test(img_list, labels,output_txt=0):
    result=''
    with tf.Session() as sess:
        correct_counter = 0
        wrong_counter = 0
        correct_dic = {}
        wrong_dic = {}
        for i in labels:
            correct_dic[i] = 0
            wrong_dic[i] = 0
        list_len = len(img_list)
        logits = sess.graph.get_tensor_by_name('final_result:0')

        result += '=====================       result      ======================\n'

        for i in img_list:
            image = tf.gfile.FastGFile(img_path + '/' + i, 'rb').read()
            prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})
            result += '===============================================================\n'
            result += i+'\n'
            result += '---------------------------------------------------------------\n'
            it_is = ''
            temp = 0

            for j in range(len(labels)):
                name = labels[j]
                score = prediction[0][j]

                if (score > temp):
                    temp = score
                    it_is = name

                result += '%s (%.2f%%) \n' % (name, score * 100)

            result += '---------------------------------------------------------------\n'

            if it_is in i:
                result += ">>result:correct\n"
                for k in labels:
                    if k in i:
                        correct_dic[k] = correct_dic[k] + 1

            else:
                result += ">>result:wrong \n"
                for k in labels:
                    if k in i:
                        wrong_dic[k] = wrong_dic[k] + 1

            # print('It is %s (%.2f%%)' % (it_is, temp * 100))
            result += '=============================================================== \n\n'
        for i in labels:
            result += '%s : (Correct:%d, Wrong: %d)   Correct rate:%.2f%% \n' % (
                i, correct_dic[i], wrong_dic[i], correct_dic[i] / (correct_dic[i] + wrong_dic[i]) * 100)
            correct_counter = correct_counter + correct_dic[i]
            wrong_counter = wrong_counter + wrong_dic[i]
        result += '=============================================================== \n'
        result += 'Data:%d (Correct:%d, Wrong: %d)   Correct rate:%.2f%%' % (
            list_len, correct_counter, wrong_counter, correct_counter / list_len * 100)

        if output_txt:
            f = open('report_' + testset + '.txt', 'w')
            f.write(result)
            f.close()

    return result



def output_log(img_list, labels):
    with tf.Session() as sess:
        correct_counter = 0
        wrong_counter = 0
        correct_dic = {}
        wrong_dic = {}
        for i in labels:
            correct_dic[i] = 0
            wrong_dic[i] = 0
        list_len = len(img_list)
        logits = sess.graph.get_tensor_by_name('final_result:0')

        print('=====================       result      ======================')

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
                for k in labels:
                    if k in i:
                        correct_dic[k] = correct_dic[k] + 1

            else:
                print(">>result:wrong")
                for k in labels:
                    if k in i:
                        wrong_dic[k] = wrong_dic[k] + 1

            # print('It is %s (%.2f%%)' % (it_is, temp * 100))
            print('=============================================================== \n')
        for i in labels:
            print('%s : (Correct:%d, Wrong: %d)   Correct rate:%.2f%%' % (
                i, correct_dic[i], wrong_dic[i], correct_dic[i] / (correct_dic[i] + wrong_dic[i]) * 100))
            correct_counter = correct_counter + correct_dic[i]
            wrong_counter = wrong_counter + wrong_dic[i]
        print('=============================================================== \n')
        print('Data:%d (Correct:%d, Wrong: %d)   Correct rate:%.2f%%' % (
            list_len, correct_counter, wrong_counter, correct_counter / list_len * 100))


def output_txt(img_list, labels):
    with tf.Session() as sess:
        correct_counter = 0
        wrong_counter = 0
        list_len = len(img_list)
        logits = sess.graph.get_tensor_by_name('final_result:0')

        f = open('report_' + testset + '.txt', 'w')
        f.write('=====================       result       ======================\n')

        for i in img_list:
            image = tf.gfile.FastGFile(img_path + '/' + i, 'rb').read()
            prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})
            f.write('===============================================================\n')
            f.write(i + '\n')
            f.write('---------------------------------------------------------------\n')
            it_is = ''
            temp = 0
            for j in range(len(labels)):
                name = labels[j]
                score = prediction[0][j]
                if (score > temp):
                    temp = score
                    it_is = name
                f.write('%s (%.2f%%) \n' % (name, score * 100))
            f.write('---------------------------------------------------------------\n')
            if it_is in i:
                f.write(">>result:correct\n")
                correct_counter = correct_counter + 1
            else:
                f.write(">>result:wrong\n")
                wrong_counter = wrong_counter + 1
            # print('It is %s (%.2f%%)' % (it_is, temp * 100))
            f.write('=============================================================== \n\n')
        f.write(testset)
        f.write('Data:%d (Correct:%d, Wrong: %d)   Correct rate:%.2f%%' % (
            list_len, correct_counter, wrong_counter, correct_counter / list_len * 100))

    f.close()


def main(_):
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]
    img_list = os.listdir(img_path)
    print(img_list)
    list_len = len(img_list)
    print(list_len)
    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    # output_log(img_list, labels)
    # output_txt(img_list, labels)
    print(result_of_test(img_list,labels,1))

    # if FLAGS.show_image:
    #     img = mpimg.imread('./workspace/flower_photos/testing34.jpg')
    #     plt.imshow(img)
    #     plt.show()


if __name__ == "__main__":
    tf.app.run()
