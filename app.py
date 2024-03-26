import torch
from PIL import Image
from RealESRGAN import RealESRGAN
import gradio as gr

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
    os.system("python inference_video.py")
    return result_video
    


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
    fn=inference_image,
    inputs=[input_video, input_model_video],
    outputs=output_video,
    title="Real-ESRGAN Pytorch",
    description="Gradio UI for Real-ESRGAN Pytorch version. To use it, simply upload your video, or click one of examples and choose the model. Read more at the links below. Please click submit only once <br><p style='text-align: center'><a href='https://arxiv.org/abs/2107.10833'>Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data</a> | <a href='https://github.com/ai-forever/Real-ESRGAN'>Github Repo</a></p>"
)

demo = gr.TabbedInterface([tab_img, tab_vid], ["Image", "Video"])



demo.launch(debug=True, show_error=True)