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
                           "./workspace/flowers_graph.pb",
                           "학습된 신경망이 저장된 위치")
tf.app.flags.DEFINE_string("output_labels",
                           "./workspace/flowers_labels.txt",
                           "학습할 레이블 데이터 파일")
tf.app.flags.DEFINE_boolean("show_image",
                            True,
                            "이미지 추론 후 이미지를 보여줍니다.")

FLAGS = tf.app.flags.FLAGS
images =[]
img_path='./workspace/flower_photos/testset'
def main(_):
    labels = [line.rstrip() for line in tf.gfile.GFile(FLAGS.output_labels)]
    img_list=os.listdir(img_path)
    print(img_list)

    print(img_list[1])

    with tf.gfile.FastGFile(FLAGS.output_graph, 'rb') as fp:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(fp.read())
        tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:

        for i in img_list:

            logits = sess.graph.get_tensor_by_name('final_result:0')
            image = tf.gfile.FastGFile(img_path+'/'+ i, 'rb').read()
            prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})

            print(i)
            for j in range(len(labels)):
                name = labels[j]
                score = prediction[0][j]
                print('%s (%.2f%%)' % (name, score * 100))

        # print(sys.argv[1])
        logits = sess.graph.get_tensor_by_name('final_result:0')
        image = tf.gfile.FastGFile('./workspace/flower_photos/testing34.jpg', 'rb').read()
        prediction = sess.run(logits, {'DecodeJpeg/contents:0': image})

    print('=== 예측 결과 ===')
    for i in range(len(labels)):
        name = labels[i]
        score = prediction[0][i]
        print('%s (%.2f%%)' % (name, score * 100))

    if FLAGS.show_image:
        img = mpimg.imread('./workspace/flower_photos/testing34.jpg')
        plt.imshow(img)
        plt.show()


if __name__ == "__main__":
    tf.app.run()


def create_list(dir):
    result=''
    return result


# def create_image_lists(image_dir, testing_percentage, validation_percentage):
#     if not gfile.Exists(image_dir):
#         print("Image directory '" + image_dir + "' not found.")
#         return None
#     result = {}
#     sub_dirs = [x[0] for x in gfile.Walk(image_dir)]
#
#
#     # The root directory comes first, so skip it.
#     is_root_dir = True
#     for sub_dir in sub_dirs:
#         if is_root_dir:
#             is_root_dir = False
#             continue
#         extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
#         file_list = []
#         dir_name = os.path.basename(sub_dir)
#         if dir_name == image_dir:
#             continue
#         print("Looking for images in '" + dir_name + "'")
#         for extension in extensions:
#             file_glob = os.path.join(image_dir, dir_name, '*.' + extension)
#             file_list.extend(gfile.Glob(file_glob))
#         if not file_list:
#             print('No files found')
#             continue
#         if len(file_list) > MAX_NUM_IMAGES_PER_CLASS:
#             print('WARNING: Folder {} has more than {} images. Some images will '
#                   'never be selected.'.format(dir_name, MAX_NUM_IMAGES_PER_CLASS))
#         label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
#         training_images = []
#         testing_images = []
#         validation_images = []
#         for file_name in file_list:
#             base_name = os.path.basename(file_name)
#
#             # hash_name = re.sub(r'_nohash_.*$', '', file_name)
#
#             # hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
#             # percentage_hash = ((int(hash_name_hashed, 16) %
#             #                     (MAX_NUM_IMAGES_PER_CLASS + 1)) *
#             #                    (100.0 / MAX_NUM_IMAGES_PER_CLASS))
#             # if percentage_hash < validation_percentage:
#             #     validation_images.append(base_name)
#             # elif percentage_hash < (testing_percentage + validation_percentage):
#             #     testing_images.append(base_name)
#             # else:
#             #     training_images.append(base_name)
#         result[label_name] = {
#             'dir': dir_name,
#             'training': training_images,
#             'testing': testing_images,
#             'validation': validation_images,
#         }
#     return result



