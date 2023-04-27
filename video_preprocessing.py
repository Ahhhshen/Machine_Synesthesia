import os
import moviepy.editor as mp
import cv2
import imageio
from PIL import Image
import numpy as np

# base folder path
base_folder_path = 'D:/OneDrive/Harvard/OneDrive - Harvard University/SCI6487_machine_aesthetics/Final_projects/Machine_Synesthesia/classifierDir'

# Input folder path containing video files

input_folder_path = os.path.join(base_folder_path, 'videos')
time_catogories = ['01_morning',  '02_noon', '03_afternoon', '04_evening']
# Output folder for image frames
output_frames_folder = os.path.join(base_folder_path, 'frames')
# Output folder for audio clips
output_audio_folder = os.path.join(base_folder_path, 'audio')
# Output folder for audios' waveforms
output_waveform_folder = os.path.join(base_folder_path, 'waveform')
# resize image to 120*120
resize = (256, 256)
# extract frames interval
interval = 1.0
interval_num = int(1/interval)

# Loop through each video file
for time_catogory in time_catogories:
    video_files = os.listdir(os.path.join(input_folder_path, time_catogory))
    for video_file in video_files:
        video_file_path = os.path.join(input_folder_path, time_catogory, video_file)
  
    # Load video clip
        video = mp.VideoFileClip(video_file_path)

    # Calculate duration of video clip
        video_duration = video.duration

    # Crop video frames to square ratio
        x_center = video.w // 2
        y_center = video.h // 2
        frame_size = min(video.w, video.h)
        video = video.crop(x_center, y_center, width=frame_size, height=frame_size)

    # Extract image frames every interval seconds
        for i in range(0, int(video_duration)*interval_num):
            frame = video.get_frame(i*interval)

            # Resize frame to 64x64
            frame = Image.fromarray(frame)
            frame = frame.resize(resize, Image.BICUBIC)
            frame = np.array(frame)

            frame_filename = f'frame_{video_file}_{i*0.1}.png'
            frame_filepath = os.path.join(output_frames_folder, time_catogory, frame_filename)
            imageio.imwrite(frame_filepath, frame)

        # Extract audio clips of interval seconds each
        for i in range(0, int(video_duration)*interval_num):
            if i + interval > video_duration:
                break
            audio_clip = video.subclip(i, i+interval).audio
            audio_filename = f'audio_{video_file}_{i}.mp3'
            audio_filepath = os.path.join(output_audio_folder, time_catogory ,audio_filename)
            audio_clip.write_audiofile(audio_filepath)
            #extract waveform from audio
            waveform = audio_clip.to_soundarray()
            #get first channel
            waveform = waveform[:,0]
            # write audio to a cvs file
            waveform_filename = f'waveform_{video_file}_{i}.csv'
            waveform_filepath = os.path.join(output_audio_folder, time_catogory, waveform_filename)
            np.savetxt(waveform_filepath, waveform, delimiter=",")

        # Close the video clip
        video.close()