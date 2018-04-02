from scipy import misc
import numpy as np
import os
import cv2
import tensorflow as tf
import detect_and_align
import locale
import shutil

locale.getdefaultlocale()
test_folder = "enrollment_data\i_t"
#test_folder = "val2017"
output_data = "detection_results"

with tf.Graph().as_default():
    with tf.Session() as sess:
        pnet, rnet, onet = detect_and_align.create_mtcnn(sess, None)
        image_names = os.listdir(os.path.expanduser(test_folder))
        image_paths = [os.path.join(test_folder, img) for img in image_names]

        for i in range(len(image_paths)):
            print(image_paths[i])
            image = misc.imread(image_paths[i])
            basename = os.path.basename(image_paths[i])
            try:
                face_patches, padded_bounding_boxes, landmarks, faces_detected = detect_and_align.align_image(image, pnet, rnet, onet)
                #print("Total Faces Detected=", len(face_patches))
                if len(faces_detected) > 0:
                    img = faces_detected[0]
                    shutil.copy(image_paths[i], os.path.join(output_data, 'images_with_face', basename))
                    cv2.imwrite(os.path.join(output_data, 'detected_faces', basename), img)
                    '''
                    cv2.imshow('face', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''
                else:
                    shutil.copy(image_paths[i], os.path.join(output_data, 'images_with_no_face', basename))
            except:
                pass
