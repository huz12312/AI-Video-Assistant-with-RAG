



import os
import subprocess

# Create output folder
os.makedirs("audio", exist_ok=True)

files = os.listdir("video")

for file in files:

    if file.endswith(".mp4"):

        input_file = os.path.join("video", file)

        name = os.path.splitext(file)[0]
        output_file = os.path.join("audio", name + ".mp3")

        cmd = [
            "ffmpeg",
            "-i", input_file,
            "-vn",
            "-ar", "44100",
            "-ac", "2",
            "-b:a", "192k",
            output_file
        ]

        print("Converting:", file)

        subprocess.run(cmd)

print("Done")

