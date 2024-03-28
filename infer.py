# import cv2
# from os.path import isfile, join
# import subprocess
# import os
# from RealESRGAN import RealESRGAN
# import torch
# import gradio as gr

# IMAGE_FORMATS = ('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')

# def inference_image(image, size):
#     global model2
#     global model4
#     global model8
#     if image is None:
#         raise gr.Error("Image not uploaded")

#     width, height = image.size
#     if width >= 5000 or height >= 5000:
#         raise gr.Error("The image is too large.")

#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()

#     if size == '2x':
#         try:
#             result = model2.predict(image.convert('RGB'))
#         except torch.cuda.OutOfMemoryError as e:
#             print(e)
#             model2 = RealESRGAN(device, scale=2)
#             model2.load_weights('weights/RealESRGAN_x2.pth', download=False)
#             result = model2.predict(image.convert('RGB'))
#     elif size == '4x':
#         try:
#             result = model4.predict(image.convert('RGB'))
#         except torch.cuda.OutOfMemoryError as e:
#             print(e)
#             model4 = RealESRGAN(device, scale=4)
#             model4.load_weights('weights/RealESRGAN_x4.pth', download=False)
#             result = model2.predict(image.convert('RGB'))
#     else:
#         try:
#             result = model8.predict(image.convert('RGB'))
#         except torch.cuda.OutOfMemoryError as e:
#             print(e)
#             model8 = RealESRGAN(device, scale=8)
#             model8.load_weights('weights/RealESRGAN_x8.pth', download=False)
#             result = model2.predict(image.convert('RGB'))

#     print(f"Frame of the Video size ({device}): {size} ... OK")
#     return result


# # assign directory
# directory = 'videos' #PATH_WITH_INPUT_VIDEOS
# zee = 0

# def convert_frames_to_video(pathIn,pathOut,fps):
#     global INPUT_DIR
#     cap = cv2.VideoCapture(f'/{INPUT_DIR}/videos/input.mp4')
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     frame_array = []
#     files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
#     #for sorting the file names properly
#     files.sort(key = lambda x: int(x[5:-4]))
#     size2 = (0,0)

#     for i in range(len(files)):
#         filename=pathIn + files[i]
#         #reading each files
#         img = cv2.imread(filename)
#         height, width, layers = img.shape
#         size = (width,height)
#         size2 = size
#         print(filename)
#         #inserting the frames into an image array
#         frame_array.append(img)
#     out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size2)
#     for i in range(len(frame_array)):
#         # writing to a image array
#         out.write(frame_array[i])
#     out.release()


# for filename in os.listdir(directory):

#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     if os.path.isfile(f):


#       print("PROCESSING :"+str(f)+"\n")
#       # Read the video from specified path

#       #video to frames
#       cam = cv2.VideoCapture(str(f))

#       try:

#           # PATH TO STORE VIDEO FRAMES
#           if not os.path.exists(f'/{INPUT_DIR}/upload/'):
#               os.makedirs(f'/{INPUT_DIR}/upload/')

#       # if not created then raise error
#       except OSError:
#           print ('Error: Creating directory of data')

#       # frame
#       currentframe = 0


#       while(True):

#           # reading from frame
#           ret,frame = cam.read()

#           if ret:
#               # if video is still left continue creating images
#               name = f'/{INPUT_DIR}/upload/frame' + str(currentframe) + '.jpg'

#               # writing the extracted images
#               cv2.imwrite(name, frame)


#                 # increasing counter so that it will
#                 # show how many frames are created
#               currentframe += 1
#               print(currentframe)
#           else:
#               #deletes all the videos you uploaded for upscaling
#               #for f in os.listdir(video_folder):
#               #  os.remove(os.path.join(video_folder, f))

#               break

#         # Release all space and windows once done
#       cam.release()
#       cv2.destroyAllWindows()

#       #apply super-resolution on all frames of a video

#       # Specify the directory path
#       all_frames_path = f"/{INPUT_DIR}/upload/"

#       # Get a list of all files in the directory
#       file_names = os.listdir(all_frames_path)

#       # process the files
#       for file_name in file_names:
#         inference_image(f"/{INPUT_DIR}/upload/{file_name}")


#       #convert super res frames to .avi
#       pathIn = f'/{INPUT_DIR}/results/restored_imgs/'

#       zee = zee+1
#       fName = "video"+str(zee)
#       filenameVid = f"{fName}.avi"

#       pathOut = f"/{INPUT_DIR}/results_videos/"+filenameVid

#       convert_frames_to_video(pathIn, pathOut, fps)


#       #convert .avi to .mp4
#       src = f'/{INPUT_DIR}/results_videos/'
#       dst = f'/{INPUT_DIR}/results_mp4_videos/'

#       for root, dirs, filenames in os.walk(src, topdown=False):
#           #print(filenames)
#           for filename in filenames:
#               print('[INFO] 1',filename)
#               try:
#                   _format = ''
#                   if ".flv" in filename.lower():
#                       _format=".flv"
#                   if ".mp4" in filename.lower():
#                       _format=".mp4"
#                   if ".avi" in filename.lower():
#                       _format=".avi"
#                   if ".mov" in filename.lower():
#                       _format=".mov"

#                   inputfile = os.path.join(root, filename)
#                   print('[INFO] 1',inputfile)
#                   outputfile = os.path.join(dst, filename.lower().replace(_format, ".mp4"))
#                   subprocess.call(['ffmpeg', '-i', inputfile, outputfile])
#               except:
#                   print("An exception occurred")

from PIL import Image
import cv2 as cv
import torch
from RealESRGAN import RealESRGAN
import tempfile
import numpy as np
import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def infer_image(img: Image.Image, size_modifier: int ) -> Image.Image:
    if img is None:
        raise Exception("Image not uploaded")
    
    width, height = img.size
    
    if width >= 5000 or height >= 5000:
        raise Exception("The image is too large.")

    model = RealESRGAN(device, scale=size_modifier)
    model.load_weights(f'weights/RealESRGAN_x{size_modifier}.pth', download=False)

    result = model.predict(img.convert('RGB'))
    print(f"Image size ({device}): {size_modifier} ... OK")
    return result

def infer_video(video_filepath: str, size_modifier: int) -> str:
    model = RealESRGAN(device, scale=size_modifier)
    model.load_weights(f'weights/RealESRGAN_x{size_modifier}.pth', download=False)

    cap = cv.VideoCapture(video_filepath)
    
    tmpfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    vid_output = tmpfile.name
    tmpfile.close()

    vid_writer = cv.VideoWriter(
        vid_output,
        fourcc=cv.VideoWriter.fourcc(*'mp4v'),
        fps=cap.get(cv.CAP_PROP_FPS),
        frameSize=(int(cap.get(cv.CAP_PROP_FRAME_WIDTH)) * size_modifier, int(cap.get(cv.CAP_PROP_FRAME_HEIGHT)) * size_modifier)
    )

    n_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # while cap.isOpened():
    for _ in tqdm.tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)

        upscaled_frame = model.predict(frame.convert('RGB'))
        
        upscaled_frame = np.array(upscaled_frame)
        upscaled_frame = cv.cvtColor(upscaled_frame, cv.COLOR_RGB2BGR)

        print(upscaled_frame.shape)

        vid_writer.write(upscaled_frame)

    vid_writer.release()

    print(f"Video file : {video_filepath}")

    return vid_output
                                

