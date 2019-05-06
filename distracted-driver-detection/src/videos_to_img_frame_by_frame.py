### ---------- Handle Command Line Arguments ----------
import argparse

hide_img = False # default is to display image after predictions

a = argparse.ArgumentParser(description="Predict the class of a given driver image.")
#a.add_argument("--image", help="path to image")
a.add_argument("--hide_img", action="store_true", help="do NOT display image on prediction termination")
args = a.parse_args()

if args.hide_img:
    hide_img = True
    
if args.video is not None:
    video_path = args.video
    
### ---------- Import Relevant Libraries ----------
import argparse
import av

video_path = args.video

container = av.open(video_path)

for packet in container.demux(video=0):
    for frame in packet.decode():
        frame.to_image().save('/home/saurabh/Development/projectX/distracted-driver-detection/dataset/images_video/frame-%04d.jpg' % frame.index)
