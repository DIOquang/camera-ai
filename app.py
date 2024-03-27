import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import gradio as gr
import os
from random import randint
import shutil

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model2 = RealESRGAN(device, scale=2)
model2.load_weights('weights/RealESRGAN_x2.pth', download=True)
model4 = RealESRGAN(device, scale=4)
model4.load_weights('weights/RealESRGAN_x4.pth', download=True)
model8 = RealESRGAN(device, scale=8)
model8.load_weights('weights/RealESRGAN_x8.pth', download=True)


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

    print(f"Image size ({device}): {size} ... OK")
    return result



def inference_video(video, size):
    _id = randint(1, 10000)
    INPUT_DIR = "tmp"
    # Check if the directory exists, if so remove it
    if os.path.exists(INPUT_DIR):
        shutil.rmtree(INPUT_DIR)
    else:
        # Create the directory, equivalent to 'mkdir -p'
        os.makedirs(INPUT_DIR, exist_ok=True)
    os.chdir(INPUT_DIR)
    
    upload_folder = 'upload'
    result_folder = 'results'
    video_folder = 'videos'
    video_result_folder = 'results_videos'
    video_mp4_result_folder = 'results_mp4_videos'
    result_restored_imgs_folder = 'restored_imgs'
    
    if os.path.isdir(upload_folder):
        print(upload_folder+" exists")
    else:
        os.makedirs(upload_folder, exist_ok=True)
    
    if os.path.isdir(video_folder):
        print(video_folder+" exists")
    else:
        os.makedirs(video_folder, exist_ok=True)
    
    if os.path.isdir(video_result_folder):
        print(video_result_folder+" exists")
    else:
        os.makedirs(video_result_folder, exist_ok=True)
        
    if os.path.isdir(video_mp4_result_folder):
        print(video_mp4_result_folder+" exists")
    else:
        os.makedirs(video_mp4_result_folder, exist_ok=True)
    
    if os.path.isdir(result_folder):
        print(result_folder+" exists")
    else:
        os.makedirs(result_folder, exist_ok=True)
    
    os.chdir("results")
    if os.path.isdir(result_restored_imgs_folder):
        print(result_restored_imgs_folder+" exists")
    else:
        os.makedirs(result_restored_imgs_folder, exist_ok=True)
    os.chdir("..")
    
    if os.path.isdir(video_folder):
        shutil.rmtree(video_folder)
    os.makedirs(video_folder, exist_ok=True)
    os.chdir("..")
    try:
        # Specify the desired output file path with the custom name and ".mp4" extension
        output_file_path = f"/{INPUT_DIR}/videos/input.mp4"

        # Save the video input to the specified file path
        with open(output_file_path, 'wb') as output_file:
            output_file.write(video)
        print(f"Video input saved as {output_file_path}")
    except Exception as e:
        print(f"Error saving video input: {str(e)}")

    os.chdir("..")
    os.system("python inference_video.py")
    return os.path.join(f'/{INPUT_DIR}/results_mp4_videos/', 'input.mp4')
    


input_image = gr.Image(type='pil', label='Input Image')
input_model_image = gr.Radio(['2x', '4x', '8x'], type="value", value="4x", label="Model Upscale/Enhance Type")
submit_image_button = gr.Button('Submit')
output_image = gr.Image(type="filepath", label="Output Image")

tab_img = gr.Interface(
    fn=inference_image,
    inputs=[input_image, input_model_image],
    outputs=output_image,
    title="Real-ESRGAN Pytorch",
    description="Gradio UI for Real-ESRGAN Pytorch version. To use it, simply upload your image, or click one of examples and choose the model. Read more at the links below. Please click submit only once <br><p style='text-align: center'><a href='https://arxiv.org/abs/2107.10833'>Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a> | <a href='https://github.com/ai-forever/Real-ESRGAN'>Github Repo</a></p>"
)

input_video = gr.Video(label='Input Video')
input_model_video = gr.Radio(['2x', '4x', '8x'], type="value", value="4x", label="Model Upscale/Enhance Type")
submit_video_button = gr.Button('Submit')
output_video = gr.Video(label='Output Video')

tab_vid = gr.Interface(
    fn=inference_video,
    inputs=[input_video, input_model_video],
    outputs=output_video,
    title="Real-ESRGAN Pytorch",
    description="Gradio UI for Real-ESRGAN Pytorch version. To use it, simply upload your video, or click one of examples and choose the model. Read more at the links below. Please click submit only once <br><p style='text-align: center'><a href='https://arxiv.org/abs/2107.10833'>Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a> | <a href='https://github.com/ai-forever/Real-ESRGAN'>Github Repo</a></p>"
)

demo = gr.TabbedInterface([tab_img, tab_vid], ["Image", "Video"])



demo.launch(debug=True, show_error=True)