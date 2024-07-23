import cv2
import glob

    #  Goal: Trích xuất các frame từ nhiều video trong một thư mục, lưu các frame này thành các file ảnh riêng biệt
def convert_videos_to_images():
    # Đặt tốc độ khung hình (frames per second) là 6
    fps = 6

    #tìm kiếm tất cả các file .mp4 trong thư mục và các thư mục con, sau đó lưu đường dẫn của chúng vào videonames_list
    folders = glob.glob('F:/Data_for_Deepfake_DL/Dataset-1/new-train/train/Real')
    videonames_list = []

    for folder in folders:
        folder_with_forward_slashes = folder.replace("\\", "/")  
        subfolders = glob.glob(folder_with_forward_slashes + "**/*.mp4")
        for mp4_file in subfolders:
            videonames_list.append(mp4_file.replace("\\", "/"))
    print('There are {} videos in Folder'.format(len(videonames_list)))

    # Sử dụng vòng lặp để lấy frame
    count = 0
    for i in range(0, len(videonames_list)):
        video_data = videonames_list[i]
        video = cv2.VideoCapture(video_data)
        success = True
        while success:
            success, image = video.read()
            name = 'F:/Data_for_Deepfake_DL/Extracted-frames/train-Real/' + str(count) + '.jpg'
            if success == True:
                #Lưu frame thành file ảnh nếu số frame chia hết cho fps (để lấy 1 frame cho mỗi 6 frame)
                if count % fps == 0:
                    cv2.imwrite(name, image)
                    print('Frame {} Extracted Successfully'.format(count))
                count += 1
            else:
                i = i + 1
            i = i + 1
        print('\n\n\nVideo {} Extracted Successfully\n\n\n'.format(video_data))

convert_videos_to_images()
