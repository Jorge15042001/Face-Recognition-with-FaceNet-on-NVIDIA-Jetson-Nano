import cv2
import numpy as np
from .calibration import getStereoRectifier
import threading
from .utils import startCameraArray, loadStereoCameraConfig, StereoConfig
import pyaudio
import wave
import time
import sys


class StereoVideoRecorder:
    def __init__(self, config_file):
        self.stereo_config = loadStereoCameraConfig(config_file)
        #  cam_center_left, cam_center_right = cam_centers
        self.f_length = min(self.stereo_config.left_camera.fpx,
                            self.stereo_config.right_camera.fpx)
        self.cams = startCameraArray(self.stereo_config.left_camera,
                                     self.stereo_config.right_camera,
                                     self.stereo_config)
        rectifier = getStereoRectifier(self.stereo_config.stereo_map_file)
        self.cams.rectifier = rectifier

        self.keep_loop = True

    def run(self):
        self.cams.start()
        while self.cams.isOpened() and self.keep_loop:
            frame_left, frame_right = self.cams.get_frames()
            succes_left, frame_left = frame_left
            succes_right, frame_right = frame_right

            if not succes_right or not succes_left:
                print("Ignoring empty camera frame.")
                continue

            terminate = self.showImages(frame_left, frame_right, )
            if terminate:
                break
        self.cams.close()
        self.keep_loop = False

    def start(self):
        threading.Thread(target=self.run).start()

    def close(self):
        self.keep_loop = False

    def showImages(self, frame_left, frame_right):
        if self.stereo_config.show_images:
            cv2.imshow("frame right", cv2.resize(
                frame_right, (np.array(frame_right.shape[:2][::-1])*1.5).astype(int)))
            cv2.imshow("frame left", cv2.resize(
                frame_left, (np.array(frame_right.shape[:2][::-1])*1.5).astype(int)))
            if cv2.waitKey(5) & 0xFF == 27:
                return True
            return False


class AudioRecorder():
    def __init__(self, config_file: str):
        path = loadStereoCameraConfig(config_file).save_path
        self.open = True
        self.rate = 44100
        self.frames_per_buffer = 44100
        self.channels = 2
        self.format = pyaudio.paInt16
        self.audio_filename = f"{path}/audio.wav"
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      #  input_device_index=0,
                                      frames_per_buffer=self.frames_per_buffer,
                                      input=True)
        self.audio_frames = []

    def record(self):
        self.stream.start_stream()
        print("recording", end="")
        while (self.open == True):
            print("... ", end="")
            data = self.stream.read(self.frames_per_buffer)
            self.audio_frames.append(data)
            if self.open == False:
                break

        print("Done")

    def stop(self):
        if self.open == True:
            self.open = False
            time.sleep(4)
            print("stoping stream")
            self.stream.stop_stream()
            print("Closing stream")
            self.stream.close()
            print("Terminating pyaudio")
            waveFile = wave.open(self.audio_filename, 'wb')
            waveFile.setnchannels(self.channels)
            waveFile.setsampwidth(self.audio.get_sample_size(self.format))
            waveFile.setframerate(self.rate)

            print("writting frames")
            waveFile.writeframes(b''.join(self.audio_frames))
            print("Fames done")
            waveFile.close()

            self.audio.terminate()
            print("mic terminated")
        pass

    def start(self):
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()


if __name__ == "__main__":
    stereo_config_file = sys.argv[1]
    video_recorder = StereoVideoRecorder(stereo_config_file)
    auduio_recorder = AudioRecorder(stereo_config_file)
    video_recorder.start()
    auduio_recorder.start()

    while video_recorder.keep_loop:
        time.sleep(1)
    auduio_recorder.stop()
