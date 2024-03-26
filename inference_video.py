import cv2
import numpy as np
import glob
from os.path import isfile, join
import subprocess
import os
import shutil
from io import BytesIO
import io
from RealESRGAN import RealESRGAN
import torch
from PIL import Image
import numpy as np


IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)


def inference_image(image, size):
    global model2
    global model4
    global model8
    if image is None:
        raise gr.Error("Image not uploaded")

    width, height = image.size
    if width >= 5000 or height >= 5000:
        raise gr.Error("The image is too large.")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if size == '2x':
        try:
            result = model2.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model2 = RealESRGAN(device, scale=2)
            model2.load_weights('weights/RealESRGAN_x2.pth', download=False)
            result = model2.predict(image.convert('RGB'))
    elif size == '4x':
        try:
            result = model4.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model4 = RealESRGAN(device, scale=4)
            model4.load_weights('weights/RealESRGAN_x4.pth', download=False)
            result = model2.predict(image.convert('RGB'))
    else:
        try:
            result = model8.predict(image.convert('RGB'))
        except torch.cuda.OutOfMemoryError as e:
            print(e)
            model8 = RealESRGAN(device, scale=8)
            model8.load_weights('weights/RealESRGAN_x8.pth', download=False)
            result = model2.predict(image.convert('RGB'))

    print(f"Frame of the Video size ({device}): {size} ... OK")
    return result

custom_name = "input.mp4"

def save_video_input(video, custom_name):
    try:
        # Specify the desired output file path with the custom name and ".mp4" extension
        output_file_path = f"/tmp/videos/{custom_name}.mp4"

        # Save the video input to the specified file path
        with open(output_file_path, 'wb') as output_file:
            output_file.write(video_input)
        print(f"Video input saved as {output_file_path}")
    except Exception as e:
        print(f"Error saving video input: {str(e)}")




# assign directory
directory = 'videos' #PATH_WITH_INPUT_VIDEOS
zee = 0

def convert_frames_to_video(pathIn,pathOut,fps):
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
    #for sorting the file names properly
    files.sort(key = lambda x: int(x[5:-4]))
    size2 = (0,0)

    for i in range(len(files)):
        filename=pathIn + files[i]
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        size2 = size
        print(filename)
        #inserting the frames into an image array
        frame_array.append(img)
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size2)
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


for filename in os.listdir(directory):

    f = os.path.join(directory, filename)
    # checking if it is a file
    if os.path.isfile(f):


      print("PROCESSING :"+str(f)+"\n")
      # Read the video from specified path

      #video to frames
      cam = cv2.VideoCapture(str(f))

      try:

          # PATH TO STORE VIDEO FRAMES
          if not os.path.exists('/tmp/upload/'):
              os.makedirs('/tmp/upload/')

      # if not created then raise error
      except OSError:
          print ('Error: Creating directory of data')

      # frame
      currentframe = 0


      while(True):

          # reading from frame
          ret,frame = cam.read()

          if ret:
              # if video is still left continue creating images
              name = '/tmp/upload/frame' + str(currentframe) + '.jpg'

              # writing the extracted images
              cv2.imwrite(name, frame)


                # increasing counter so that it will
                # show how many frames are created
              currentframe += 1
              print(currentframe)
          else:
              #deletes all the videos you uploaded for upscaling
              #for f in os.listdir(video_folder):
              #  os.remove(os.path.join(video_folder, f))

              break

        # Release all space and windows once done
      cam.release()
      cv2.destroyAllWindows()

      #apply super-resolution on all frames of a video

      # Specify the directory path
      all_frames_path = "/tmp/upload/"

      # Get a list of all files in the directory
      file_names = os.listdir(all_frames_path)

      # process the files
      for file_name in file_names:
        inference_image(f"/tmp/upload/{file_name}")


      #convert super res frames to .avi
      pathIn = '/tmp/results/restored_imgs/'

      zee = zee+1
      fName = "video"+str(zee)
      filenameVid = f"{fName}.avi"

      pathOut = "/tmp/results_videos/"+filenameVid

      convert_frames_to_video(pathIn, pathOut, fps)


      #convert .avi to .mp4
      src = '/tmp/results_videos/'
      dst = '/tmp/results_mp4_videos/'

      for root, dirs, filenames in os.walk(src, topdown=False):
          #print(filenames)
          for filename in filenames:
              print('[INFO] 1',filename)
              try:
                  _format = ''
                  if ".flv" in filename.lower():
                      _format=".flv"
                  if ".mp4" in filename.lower():
                      _format=".mp4"
                  if ".avi" in filename.lower():
                      _format=".avi"
                  if ".mov" in filename.lower():
                      _format=".mov"

                  inputfile = os.path.join(root, filename)
                  print('[INFO] 1',inputfile)
                  outputfile = os.path.join(dst, filename.lower().replace(_format, ".mp4"))
                  subprocess.call(['ffmpeg', '-i', inputfile, outputfile])
              except:
                  print("An exception occurred")