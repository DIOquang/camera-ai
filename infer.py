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
                                

