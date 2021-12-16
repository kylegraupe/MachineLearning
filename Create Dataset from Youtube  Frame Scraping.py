"""The purpose of this script is to create a dataset of images easily by scraping Youtube videos and extracting
frames from them."""

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


if __name__ == '__main__':
    """PART 1: Run this part first"""
    url = 'https://www.youtube.com/watch?v=FReibAoQaRA'  # Enter the URL of the video here.
    video = get_youtube_mp4(url)
    frame_spacing = 100
    print("========================================")
    print("Video Length: " + str(video.length) + "\n")  # Length of the video in seconds
    print("========================================")
    print(video.streams.filter(file_extension="mp4"))  # Shows list of all available streams.
    print("========================================")
    video.streams.get_by_itag(18).download()  # Download individual stream by the "itag"
    print(video.title)
    title = str(video.title)
    file = title + ".mp4"
    fe = FrameExtractor(file)  # Video mp4 file you want to extract
    # print(fe.get_video_duration())

    fe.get_n_images(every_x_frame=frame_spacing)
    fe.extract_frames(every_x_frame=frame_spacing,
                      img_name='dog',
                      dest_path='dog_images_2')

    show_image('dog_images_2/dog_0.jpg')
