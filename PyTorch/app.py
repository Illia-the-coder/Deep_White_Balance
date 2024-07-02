import argparse
import logging
import os
import torch
from PIL import Image
from arch import deep_wb_model
import utilities.utils as utls
from utilities.deepWB import deep_wb
import arch.splitNetworks as splitter
from arch import deep_wb_single_task
import gradio as gr

def load_models(task, model_dir, device):
    print(f"Loading models from {model_dir} for task {task}")
    
    if task == 'all':
        if os.path.exists(os.path.join(model_dir, 'net_awb.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_t.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_s.pth')):
            print("Found all required models for 'all' task")
            # load awb net
            net_awb = deep_wb_single_task.deepWBnet()
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
            # load tungsten net
            net_t = deep_wb_single_task.deepWBnet()
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            # load shade net
            net_s = deep_wb_single_task.deepWBnet()
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            print("Found single combined model for 'all' task")
            net = deep_wb_model.deepWBNet()
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            net_awb, net_t, net_s = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
        else:
            raise Exception('Model not found!')
        return net_awb, net_t, net_s
    elif task == 'editing':
        if os.path.exists(os.path.join(model_dir, 'net_t.pth')) and \
                os.path.exists(os.path.join(model_dir, 'net_s.pth')):
            print("Found models for 'editing' task")
            # load tungsten net
            net_t = deep_wb_single_task.deepWBnet()
            net_t.to(device=device)
            net_t.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_t.pth'), map_location=device))
            net_t.eval()
            # load shade net
            net_s = deep_wb_single_task.deepWBnet()
            net_s.to(device=device)
            net_s.load_state_dict(
                torch.load(os.path.join(model_dir, 'net_s.pth'), map_location=device))
            net_s.eval()
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            print("Found single combined model for 'editing' task")
            net = deep_wb_model.deepWBNet()
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            _, net_t, net_s = splitter.splitNetworks(net)
            net_t.to(device=device)
            net_t.eval()
            net_s.to(device=device)
            net_s.eval()
        else:
            raise Exception('Model not found!')
        return None, net_t, net_s
    elif task == 'awb':
        if os.path.exists(os.path.join(model_dir, 'net_awb.pth')):
            print("Found model for 'awb' task")
            # load awb net
            net_awb = deep_wb_single_task.deepWBnet()
            net_awb.to(device=device)
            net_awb.load_state_dict(torch.load(os.path.join(model_dir, 'net_awb.pth'),
                                               map_location=device))
            net_awb.eval()
        elif os.path.exists(os.path.join(model_dir, 'net.pth')):
            print("Found single combined model for 'awb' task")
            net = deep_wb_model.deepWBNet()
            net.load_state_dict(torch.load(os.path.join(model_dir, 'net.pth')))
            net_awb, _, _ = splitter.splitNetworks(net)
            net_awb.to(device=device)
            net_awb.eval()
        else:
            raise Exception('Model not found!')
        return net_awb, None, None
    else:
        raise Exception("Wrong task! Task should be: 'AWB', 'editing', or 'all'")

def process_image(image_path, task='all', target_color_temp=None, model_dir='./models', device='cuda'):
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    net_awb, net_t, net_s = load_models(task, model_dir, device)
    img = Image.open(image_path)
    if task == 'all':  # awb and editing tasks
        out_awb, out_t, out_s = deep_wb(img, task=task, net_awb=net_awb, net_s=net_s, net_t=net_t, device=device, s=656)
        out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
        return [utls.to_image(out_awb), utls.to_image(out_t), utls.to_image(out_s), utls.to_image(out_f), utls.to_image(out_d), utls.to_image(out_c)]
    elif task == 'awb':  # awb task
        out_awb = deep_wb(img, task=task, net_awb=net_awb, device=device, s=656)
        return utls.to_image(out_awb)
    else:  # editing
        out_t, out_s = deep_wb(img, task=task, net_s=net_s, net_t=net_t, device=device, s=656)
        if target_color_temp:
            out = utls.colorTempInterpolate_w_target(out_t, out_s, target_color_temp)
            return utls.to_image(out)
        else:
            out_f, out_d, out_c = utls.colorTempInterpolate(out_t, out_s)
            return [utls.to_image(out_t), utls.to_image(out_s), utls.to_image(out_f), utls.to_image(out_d), utls.to_image(out_c)]

def wb_gradio_interface(image_path, task, target_color_temp, device):
    if task == 'editing' and target_color_temp is None:
        return "Please specify a target color temperature for the editing task."
    if task == 'all':
        results = process_image(image_path, task=task, target_color_temp=target_color_temp, device=device)
        return results
    else:
        result = process_image(image_path, task=task, target_color_temp=target_color_temp, device=device)
        return result

# Gradio Interface
iface = gr.Interface(
    fn=wb_gradio_interface,
    inputs=[
        gr.Image(type="filepath", label="Input Image"),
        gr.Radio(choices=["all"], label="Task", value="all"),
        gr.Slider(2850, 7500, step=50, label="Target Color Temperature"),
        gr.Radio(choices=["cuda", "cpu"], label="Device", value="cuda"),
    ],
    outputs=gr.Gallery(label="Output Images"),
    title="Deep White Balance Editing",
    description="Upload an image and choose a task to apply white balance adjustments using deep learning models.",
)

if __name__ == "__main__":
    iface.launch(share=True)
