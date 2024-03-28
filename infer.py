from PIL import Image
import cv2
import torch
from RealESRGAN import RealESRGAN
import tempfile
import numpy as np
import tqdm
import pydub
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, AudioFileClip

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
    
    # Extract audio from the original video file
    audio = AudioSegment.from_file(video_filepath, format="mp4")
    audio_array = np.array(audio.get_array_of_samples())

    # Create a VideoCapture object for the video file
    cap = cv2.VideoCapture(video_filepath)

    # Create a temporary file for the output video
    tmpfile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    vid_output = tmpfile.name
    tmpfile.close()

    # Create a VideoWriter object for the output video
    vid_writer = cv2.VideoWriter(
        vid_output,
        fourcc=cv2.VideoWriter.fourcc(*'mp4v'),
        fps=cap.get(cv2.CAP_PROP_FPS),
        frameSize=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * size_modifier, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) * size_modifier)
    )

    # Process each frame of the video and write it to the output video
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(n_frames)):
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB and feed it to the model
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        upscaled_frame = model.predict(frame.convert('RGB'))

        # Convert the upscaled frame back to BGR and write it to the output video
        upscaled_frame = np.array(upscaled_frame)
        upscaled_frame = cv2.cvtColor(upscaled_frame, cv2.COLOR_RGB2BGR)

        # Write the upscaled frame to the output video
        vid_writer.write(upscaled_frame)

    # Release the VideoCapture and VideoWriter objects
    cap.release()
    vid_writer.release()

    # Create a new VideoFileClip object from the output video
    output_clip = VideoFileClip(vid_output)

    # Add the audio back to the output video
    audio_clip = AudioFileClip(f"{video_filepath.split('.')[0]}.wav", fps=output_clip.fps)
    output_clip = output_clip.set_audio(audio_clip)

    # Save the output video to a new file
    output_clip.write_videofile(f'output_{video_filepath}')

    return f'output_{video_filepath}'
                                
