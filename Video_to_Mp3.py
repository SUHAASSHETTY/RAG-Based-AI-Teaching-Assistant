# Converts the videos to Mp3
import os
import subprocess

files = os.listdir("videos")
for file in files:
    if file.endswith('.mp4'):
        tutorial_number = file.split(".")[0].strip()
        file_name = file.split(".", 1)[1].replace(".mp4", "").strip()
        print(tutorial_number, file_name)
        subprocess.run(["ffmpeg", "-i", f"videos/{file}",f"audios/{tutorial_number}_{file_name}.mp3"])