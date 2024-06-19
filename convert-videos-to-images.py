import cv2
import glob

def convert_videos_to_images():
    # Đọc video từ folder dataset và lưu frame vào folder extracted-frames
    # 6 giây thì lấy 1 frame
    fps = 6

    # Tổng video có trong folder dataset
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
            name = 'F:/Data_for_Deepfake_DL/Extracted-frames/train-real' + str(count) + '.jpg'
            if success == True:
                if count % fps == 0:
                    cv2.imwrite(name, image)
                    print('Frame {} Extracted Successfully'.format(count))
                count += 1
            else:
                i = i + 1
            i = i + 1
        print('\n\n\nVideo {} Extracted Successfully\n\n\n'.format(video_data))

convert_videos_to_images()
