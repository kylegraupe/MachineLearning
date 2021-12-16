"""The purpose of this script is to create a dataset of images easily by scraping YouTube videos and extracting
frames from them. Run this script on several YouTube videos to collect your dataset before processing.
Make sure to select a video resolution that is aligned with the objective of your task."""

# Import Libraries
from pytube import YouTube
import os
import math
import datetime
import matplotlib.pyplot as plt
from cv2 import cv2


class FrameExtractor():
    """
    Class used for extracting frames from a video file.
    """

    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        duration = self.n_frames / self.fps  # Number of frames divided by frames per second
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame):
        n_images = math.floor(self.n_frames / every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext='.jpg'):
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)

        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the following directory: {dest_path}')

        frame_cnt = 0
        img_cnt = 0

        while self.vid_cap.isOpened():

            success, image = self.vid_cap.read()

            if not success:
                break

            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                cv2.imwrite(img_path, image)
                img_cnt += 1

            frame_cnt += 1

        self.vid_cap.release()
        cv2.destroyAllWindows()


def show_image(path):
    image = cv2.imread(path)
    plt.imshow(image)
    plt.show()


def get_youtube_mp4(vid_url_):
    video_ = YouTube(vid_url_)
    return video_


def url_list(input_list_, image_class_, frame_spacing_):
    for i in input_list_:
        j = 0
        video = get_youtube_mp4(i)
        video.streams.get_by_itag(18).download()  # Download individual stream by the "itag". 340p is currently selected

        title = str(video.title)
        file = title + ".mp4"

        fe = FrameExtractor(file)  # Video mp4 file you want to extract
        fe.get_n_images(every_x_frame=frame_spacing_)
        fe.extract_frames(every_x_frame=frame_spacing_,
                          img_name=image_class_,
                          dest_path=image_class_ + '_images_' + str(j))  # Change image names and destinations

        j += 1


def inputs():
    # creating an empty list
    url_list_input = []

    # number of elements as input
    n = int(input("Enter number of URLs to be input: "))

    # iterating till the range
    for i in range(0, n):
        ele = str(input("URL " + str(i+1) + ": "))

        url_list_input.append(ele)  # adding the element

    image_class_ = str(input("Enter the name you would like to use for the image class: "))
    spacing_ = int(input("Enter the number of frames between images you would like to use: "))

    return url_list_input, image_class_, spacing_


if __name__ == '__main__':
    """Upon running the script, the user will be asked to input the number of URLs he or she will be using to 
    generate the dataset. The user will then be prompted to separately input each video's URL. Then the script will
    ask for the class of the image. This class will be used to name of the folder containing the images and 
    to name the individual images. The script will ask how many frames in between captures he or she would like (the
    higher this number, the lower the number of images captured). A folder will be created in the current working
    directory with the frames captured from the video. Enjoy!"""

    input_list, image_class, spacing = inputs()
    url_list(input_list, image_class, spacing)
