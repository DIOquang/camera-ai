import cv2
import numpy as np
import glob
from os.path import isfile, join
import subprocess
from IPython.display import clear_output
import os
from google.colab import files
import shutil
from io import BytesIO
import io

IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')


model_scale = "2" #@param ["2", "4", "8"] {allow-input: false}

model = RealESRGAN(device, scale=int(model_scale))
model.load_weights(f'weights/RealESRGAN_x{model_scale}.pth', download=False)


def process_input(filename):
  result_image_path = os.path.join('results/restored_imgs', os.path.basename(filename))
  image = Image.open(filename).convert('RGB')
  sr_image = model.predict(np.array(image))
  sr_image.save(result_image_path)
  print(f'Finished! Frame of the Video saved to {result_image_path}')


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
          if not os.path.exists('upload'):
              os.makedirs('upload')

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
              name = 'upload/frame' + str(currentframe) + '.jpg'

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
      all_frames_path = "upload"

      # Get a list of all files in the directory
      file_names = os.listdir(all_frames_path)

      # process the files
      for file_name in file_names:
        process_input(f"upload/{file_name}")


      #convert super res frames to .avi
      pathIn = 'results/restored_imgs/'

      zee = zee+1
      fName = "video"+str(zee)
      filenameVid = f"{fName}.avi"

      pathOut = "results_videos/"+filenameVid

      fps = 25.0 #change this to FPS of your source video

      convert_frames_to_video(pathIn, pathOut, fps)


      #convert .avi to .mp4
      src = 'results_videos/'
      dst = 'results_mp4_videos/'

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