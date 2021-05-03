import numpy as np
import pandas as pd
from PIL import Image
import cv2
import random, os, torch, pafy
import warnings, re
import argparse
import progressbar

from ImageObjectDetection import Detector

VIDEO_RES_ALLOWED = ['1280x720', '1920x1080']
STREAM_MIN_FILE_SIZE = 5 * (2 ** 20)  # 5MB
DEFAULT_OUTPUT_PATH = "datasets" + os.path.sep
START_FRAME_TIME = 3
FRAME_INTERVAL = 30
SCORE_THRES = 0.8


class Player_BB_Detector(object):
    def __init__(self, args):
        use_cuda = torch.cuda.is_available()
        if not use_cuda:
            print("Warning: Running in cpu mode!")
        self.is_youtube_video = True if args.YOUTUBE_ID else False
        if self.is_youtube_video:
            self.YOUTUBE_ID = args.YOUTUBE_ID
            if len(self.YOUTUBE_ID) > 11:
                self.YOUTUBE_ID = self.YOUTUBE_ID[-11:]
        else:
            self.LOCAL_VID = args.LOCAL_VID

        self.vid_info = {}
        self.OUTPUT_PATH = args.output_path
        self.FRAME_SAMPLE_RATE = args.frame_interval
        self.START_FRAME_TIMESTAMP = args.start_time * 1000  # move to milliseconds
        self.END_FRAME_TIMESTAMP = args.end_time * 1000  # move to milliseconds
        self.vdo = cv2.VideoCapture()
        self.detector = Detector(score_thres=args.score_thres, resize_flag=args.resize_flag)
        self.df = self.create_dataframe()

    def __enter__(self):
        if self.is_youtube_video:
            try:
                url = "https://www.youtube.com/watch?v=" + self.YOUTUBE_ID
                self.download_YouTube_video(url)
            except Exception as e:
                print("Downloading from YouTube failed, try another video url")
                raise e
        else:
            self.vid_info['path'], self.vid_info['filename'] = os.path.split(self.LOCAL_VID)
            self.vid_info['path'] += os.path.sep
            assert os.path.isfile(self.vid_info['path'] + self.vid_info['filename'])
        self.vdo.open(self.vid_info['path'] + self.vid_info['filename'])
        assert self.vdo.isOpened()
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.vdo.release()
        if self.is_youtube_video and os.path.isfile(self.vid_info['path'] + self.vid_info['filename']):
            os.remove(self.vid_info['path'] + self.vid_info['filename'])
            print("Removed temporary downloaded video from this machine")
        self.df.to_csv(self.vid_info['path'] + 'detections.csv', index=False)
        if exc_type:
            print(exc_type, exc_value, exc_traceback)

    def run(self):
        idx_frame, frames_processed = 0, 0
        total_frames_to_analyse = self.total_frames_to_analyse()
        print("Starting to process this video")
        with progressbar.ProgressBar(max_val=int(total_frames_to_analyse)) as bar:
            while self.vdo.grab():
                idx_frame += 1
                cur_timestamp = self.vdo.get(cv2.CAP_PROP_POS_MSEC)
                if idx_frame % self.FRAME_SAMPLE_RATE or cur_timestamp < self.START_FRAME_TIMESTAMP:
                    continue
                if 0 < self.END_FRAME_TIMESTAMP < cur_timestamp:
                    break
                _, ori_im = self.vdo.retrieve()
                im = cv2.cvtColor(ori_im, cv2.COLOR_BGR2RGB)
                bar.update(min(frames_processed, total_frames_to_analyse))
                detected_bbs_imgs, detected_bbs_info, detected_bbs_kp = self.detector(im)
                self.save_data(detected_bbs_imgs, detected_bbs_info, detected_bbs_kp,
                               int(self.vdo.get(cv2.CAP_PROP_POS_FRAMES)), cur_timestamp)
                frames_processed += 1
            bar.update(total_frames_to_analyse)

        print("Finished analysing the video")

    def download_YouTube_video(self, url, download_best_quality=True):
        video = pafy.new(url)
        filtered_streams = [x for x in video.allstreams if x.extension == 'mp4' and x.resolution in VIDEO_RES_ALLOWED
                            and x.get_filesize() > STREAM_MIN_FILE_SIZE]
        filtered_streams.sort(key=lambda x: int(x.resolution.split(sep="x")[0]), reverse=True)
        assert len(filtered_streams) > 0, "Error: No streams to download, try another YouTube video"
        if download_best_quality:
            stream = filtered_streams[0]
        else:
            stream = filtered_streams[random.randrange(len(filtered_streams))]
        # strip illegal chars from filename
        dirname = re.sub('[^\w\-_\. ]', '_', stream.title + str(stream))
        self.vid_info['path'] = self.OUTPUT_PATH + os.path.sep + dirname + os.path.sep
        os.makedirs(self.vid_info['path'], exist_ok=True)
        self.vid_info['filename'] = stream.filename
        print("Temporarily downloading YouTube video - " + video.title + " " + stream.resolution)
        stream.download(filepath=self.vid_info['path'] + self.vid_info['filename'], quiet=False)
        self.vid_info['video_info'] = str(stream)
        self.vid_info['videoid'] = video.videoid

    def save_data(self, detected_bbs_imgs, detected_bbs_info, detected_bbs_kp, frame_id, frame_timestamp):
        for idx, (img, bb_info, kp) in enumerate(zip(detected_bbs_imgs, detected_bbs_info, detected_bbs_kp)):
            x, y, h, w = bb_info
            left_shoulder, right_shoulder, left_hip, right_hip = self.get_torso_keypoints(kp, x, y)
            bb_id = (str(frame_id).zfill(7) + "_" + str(idx).zfill(3))
            self.df = self.df.append({'YouTube_ID': self.vid_info['videoid'],
                                      'frame_num': frame_id,
                                      'timestamp_ms': frame_timestamp,
                                      'bb_id': bb_id,
                                      'x': x, 'y': y, 'w': h, 'h': w,
                                      'x_norm': x / self.im_width,
                                      'y_norm': y / self.im_height,
                                      'w_norm': w / self.im_width,
                                      'h_norm': h / self.im_height,
                                      'l_shoulder_x': left_shoulder[0],
                                      'l_shoulder_y': left_shoulder[1],
                                      'r_shoulder_x': right_shoulder[0],
                                      'r_shoulder_y': right_shoulder[1],
                                      'l_hip_x': left_hip[0],
                                      'l_hip_y': left_hip[1],
                                      'r_hip_x': right_hip[0],
                                      'r_hip_y': right_hip[1]},
                                     ignore_index=True)
            filename = self.vid_info['videoid'] + "_" + bb_id + '.jpeg'
            Image.fromarray(img, 'RGB').save(self.vid_info['path'] + filename)

    def total_frames_to_analyse(self):
        total_frames = self.vdo.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = self.vdo.get(cv2.CAP_PROP_FPS)
        end_time_ms = self.END_FRAME_TIMESTAMP if self.END_FRAME_TIMESTAMP > 0 else (total_frames / fps) * 1000
        return (((end_time_ms - self.START_FRAME_TIMESTAMP) / 1000) * fps) // self.FRAME_SAMPLE_RATE

    @staticmethod
    def get_torso_keypoints(keypoints, offset_x=0, offset_y=0):
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.detach().cpu().numpy().astype(int)
        left_shoulder = [int(keypoints[5][0]) - offset_x, int(keypoints[5][1]) - offset_y]
        right_shoulder = [int(keypoints[6][0]) - offset_x, int(keypoints[6][1]) - offset_y]
        if left_shoulder[0] > right_shoulder[0]:
            left_shoulder, right_shoulder = right_shoulder, left_shoulder
        left_hip = [int(keypoints[11][0]) - offset_x, int(keypoints[11][1]) - offset_y]
        right_hip = [int(keypoints[12][0]) - offset_x, int(keypoints[12][1]) - offset_y]
        if left_hip[0] > right_hip[0]:
            left_hip, right_hip = right_hip, left_hip
        return left_shoulder, right_shoulder, left_hip, right_hip

    @staticmethod
    def create_dataframe():
        dtypes = np.dtype([
            ('YouTube_ID', str),
            ('frame_num', int),
            ('timestamp_ms', float),
            ('bb_id', str),
            ('x', int),
            ('y', int),
            ('w', int),
            ('h', int),
            ('x_norm', int),
            ('y_norm', int),
            ('w_norm', int),
            ('h_norm', int),
            ('l_shoulder_x', int),
            ('l_shoulder_y', int),
            ('r_shoulder_x', int),
            ('r_shoulder_y', int),
            ('l_hip_x', int),
            ('l_hip_y', int),
            ('r_hip_x', int),
            ('r_hip_y', int),
            ('digit_one_bb_x', int),
            ('digit_one_bb_y', int),
            ('digit_one_bb_w', int),
            ('digit_one_bb_h', int),
            ('digit_one_tag', int),
            ('digit_two_bb_x', int),
            ('digit_two_bb_y', int),
            ('digit_two_bb_w', int),
            ('digit_two_bb_h', int),
            ('digit_two_tag', int),
            ('tagged', bool)
        ])
        data = np.empty(0, dtype=dtypes)
        return pd.DataFrame(data)


def parse_args():
    parser = argparse.ArgumentParser(description="Detect potential bounding boxes of players in a given video")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--YOUTUBE_ID", type=str, default=None)
    group.add_argument("--LOCAL_VID", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH,
                        help="Path for the created directory with detections")
    parser.add_argument("--score_thres", type=float, default=SCORE_THRES,
                        help="Detector bounding box score threshold to include it in the output. range between 0 to 1")
    parser.add_argument("--frame_interval", type=int, default=FRAME_INTERVAL,
                        help="Detection frame rate interval")
    parser.add_argument("-s", "--start_time", type=float, default=START_FRAME_TIME,
                        help="Start detection from this second (float)")
    parser.add_argument("-e", "--end_time", type=float, default=-1.0, help="End detection from this second (float)")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True, help="Run on cpu instead of gpu")
    parser.add_argument("--resize_video", dest="resize_flag", action="store_true", default=False,
                        help="Resize video down to 512x512 pixels for a faster run")
    return parser.parse_args()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    args = parse_args()
    assert args.end_time < 0 or args.end_time > args.start_time

    with Player_BB_Detector(args) as player_BB_detector:
        player_BB_detector.run()
