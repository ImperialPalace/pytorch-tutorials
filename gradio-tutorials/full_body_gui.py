import gradio as gr
import os

from ui_common import create_refresh_button

def read_data(path):
    data = []
    files =  os.listdir(path)
    
    for file in files:
       data.append(os.path.join(path, file))
    return data

example_path = os.path.join(os.path.dirname(__file__), 'zjai_examples')
model_hd = os.path.join(example_path, 'model/half-body/model_1.png')
garment_hd = os.path.join(example_path, 'garment/upper-body/00/03244_00.jpg')
model_dc = os.path.join(example_path, 'model/body/model_8.png')
garment_dc = os.path.join(example_path, 'garment/upper-body/1/048554_1.jpg')

def process_dc(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    
    return None

def full_body_tab(
    headless=False,
):

    with gr.Row():
        gr.Markdown("## 全身")
    with gr.Row():
        gr.Markdown("***试衣间***")
    with gr.Row():
        with gr.Column():
            vton_img_dc = gr.Image(label="模特", sources='upload', type="filepath", height=384, value=model_dc)
            model_body_imgs = read_data(os.path.join(example_path,"model/body"))
            
            example = gr.Examples(
                label="模特列表(全身)",
                inputs=vton_img_dc,
                examples_per_page=7,
                examples=model_body_imgs)
            
        with gr.Column():
            garm_img_dc = gr.Image(label="服装", sources='upload', type="filepath", height=384, value=garment_dc)
            category_dc = gr.Dropdown(label="Garment category (important option!!!)", choices=["Upper-body", "Lower-body"], value="Upper-body")
            garment_upper_body_imgs = read_data(os.path.join(example_path,"garment/upper-body/1"))
            garment_lower_body_imgs = read_data(os.path.join(example_path,"garment/lower-body"))
            
            example = gr.Examples(
                label="服装列表(上半身)",
                inputs=garm_img_dc,
                examples_per_page=7,
                examples=garment_upper_body_imgs)
            example = gr.Examples(
                label="服装列表(下半身)",
                inputs=garm_img_dc,
                examples_per_page=7,
                examples=garment_lower_body_imgs)
            
        with gr.Column():
            with gr.Row():
                run_button = gr.Button(value="运行",variant="primary")
                
            result_gallery_dc = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)   
    with gr.Column():
        # run_button_dc = gr.Button(value="运行")
        n_samples_dc = gr.Slider(label="生成的图片数量", minimum=1, maximum=4, value=1, step=1)
        n_steps_dc = gr.Slider(label="迭代步数", minimum=20, maximum=40, value=20, step=1)
        # scale_dc = gr.Slider(label="Scale", minimum=1.0, maximum=12.0, value=5.0, step=0.1)
        image_scale_dc = gr.Slider(label="引导参数", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed_dc = gr.Slider(label="随机种子", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips_dc = [vton_img_dc, garm_img_dc, category_dc, n_samples_dc, n_steps_dc, image_scale_dc, seed_dc]
    run_button.click(fn=process_dc, inputs=ips_dc, outputs=[result_gallery_dc])
                 
    return (
    )