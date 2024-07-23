import time
import cv2
import glob
import numpy as np
import os

# Bắt đầu đo thời gian
start_time = time.time()

#Tải mô hình phát hiện khuôn mặt:
net = cv2.dnn.readNetFromCaffe('./DNN_face_detector/deploy.prototxt', './DNN_face_detector/res10_300x300_ssd_iter_140000.caffemodel')

#Lấy danh sách các file ảnh
path = glob.glob('F:/Data_for_Deepfake_DL/Dataset-2/test/Fake/*.jpg')
saved = 0

#Vòng lặp qua từng file ảnh:
for file in path:
    saved += 1
    #Đọc ảnh và lấy kích thước:
    img = cv2.imread(file)
    try:
        (h,w) = img.shape[:2]
    except Exception as e:
        continue
    #Tiền xử lý ảnh cho mô hình DNN:
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))
    #Đưa ảnh qua mạng neural để phát hiện khuôn mặt:

    net.setInput(blob)
    detections = net.forward()
    #Xử lý kết quả phát hiện, Chọn khuôn mặt có độ tin cậy cao nhất và lớn hơn 0.55.


    if len(detections) > 0:
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]
        if confidence > 0.55:
            box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype('int')
            #Cắt và lưu khuôn mặt
            try:
                face = img[startY:endY, startX:endX]
                p = os.path.sep.join(['F:/Data_for_Deepfake_DL/Dataset-final/test/Fake', "{}.jpg".format(saved)])
                cv2.imwrite(p, face)
                print("Đã lưu vào {}".format(p))
            except Exception as e:
                continue
#Kết thúc đo thời gian và in ra:
end_time = time.time()
elapsed_time = end_time - start_time
print("elapsed time:{0}".format(elapsed_time) + "[sec]")    
    