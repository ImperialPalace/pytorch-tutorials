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

def process_hd(vton_img, garm_img, n_samples, n_steps, image_scale, seed):
    
    return None

def upper_body_tab(
    train_data_dir_input=gr.Textbox(),
    reg_data_dir_input=gr.Textbox(),
    output_dir_input=gr.Textbox(),
    logging_dir_input=gr.Textbox(),
    headless=False,
):
    dummy_db_true = gr.Label(value=True, visible=False)
    dummy_db_false = gr.Label(value=False, visible=False)
    dummy_headless = gr.Label(value=headless, visible=False)

    # with gr.Tab('Training'):
    #     gr.Markdown(
    #         'Train a custom model using kohya train network LoRA python code...'
    #     )
    with gr.Row():
        gr.Markdown("## Full-body")
    with gr.Row():
        gr.Markdown("***Support upper-body/lower-body/dresses; garment category must be paired!!!***")
    with gr.Row():
        with gr.Column():
            vton_img = gr.Image(label="模特", sources='upload', type="filepath", height=384, value=model_hd)
            model_imgs = read_data(os.path.join(example_path,"model/half-body"))
            example = gr.Examples(
                label="模特列表",
                inputs=vton_img,
                examples_per_page=14,
                examples=model_imgs)
            
            # create_refresh_button(example, read_data, os.path.join(example_path,"model/half-body"), "refresh_train_embedding_name")
            # upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video"], file_count="multiple")
            # upload_button.upload(upload_file, upload_button, file_output)
    
        with gr.Column():
            garm_img = gr.Image(label="服装", sources='upload', type="filepath", height=384, value=garment_hd)
            garment_imgs = read_data(os.path.join(example_path,'garment/upper-body/00'))
            example = gr.Examples(
                label="服装列表",
                inputs=garm_img,
                examples_per_page=14,
                examples=garment_imgs)
        with gr.Column():
            with gr.Row():
                run_button = gr.Button(value="运行",variant="primary")
            
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery", preview=True, scale=1)
    with gr.Column():
        
        n_samples = gr.Slider(label="生成的图片数量", minimum=1, maximum=4, value=1, step=1)
        n_steps = gr.Slider(label="迭代步数", minimum=20, maximum=40, value=20, step=1)
        # scale = gr.Slider(label="Scale", minimum=1.0, maximum=12.0, value=5.0, step=0.1)
        image_scale = gr.Slider(label="引导参数", minimum=1.0, maximum=5.0, value=2.0, step=0.1)
        seed = gr.Slider(label="随机种子", minimum=-1, maximum=2147483647, step=1, value=-1)
        
    ips = [vton_img, garm_img, n_samples, n_steps, image_scale, seed]
    run_button.click(fn=process_hd, inputs=ips, outputs=[result_gallery])
                 
    return (
    )