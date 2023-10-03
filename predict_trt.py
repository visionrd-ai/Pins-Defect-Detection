import argparse
import logging
import os
import numpy as np
from PIL import Image
import common
import tensorrt as trt
import cv2
import time

TRT_LOGGER = trt.Logger()

def get_engine(onnx_file_path, engine_file_path=""):
    def build_engine():
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH
        ) as network, builder.create_builder_config() as config, trt.OnnxParser(
            network, TRT_LOGGER
        ) as parser, trt.Runtime(
            TRT_LOGGER
        ) as runtime:
            # config.max_workspace_size = 1 << 4  # 256MiB
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 4) # 1 MiB
            config.set_flag(trt.BuilderFlag.INT8)
            builder.max_batch_size = 1
            if not os.path.exists(onnx_file_path):
                print(
                    "ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.".format(onnx_file_path)
                )
                exit(0)
            print("Loading ONNX file from path {}...".format(onnx_file_path))
            with open(onnx_file_path, "rb") as model:
                print("Beginning ONNX file parsing")
                if not parser.parse(model.read()):
                    print("ERROR: Failed to parse the ONNX file.")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            network.get_input(0).shape = [1, 3, 320, 800]
            print("Completed parsing of ONNX file")
            print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    if os.path.exists(engine_file_path):
        with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()
    
def preprocess_pins(img, pinbox_color, hybrid=False):
    scale=1
    height, width = img.shape[:2]
    newW, newH = int(scale * width), int(scale * height)
    assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'

    if pinbox_color == 'black0' and hybrid==True:
        size = (174, 361)
    elif pinbox_color =='black0' and hybrid==False:
        size = (139, 352)
    elif pinbox_color == 'white0' and hybrid==True:
        size = (179, 318)
    elif pinbox_color == 'white0' and hybrid==False:
        size = (189, 326)
    elif pinbox_color == 'white1':
        size = (245, 384)
    elif pinbox_color == 'gray':
        size = (245, 313)

    img = cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)#pil_img.resize((newW, newH), resample=Image.BICUBIC)

    if img.ndim == 2:
        img = img[np.newaxis, ...]
    else:
        img = img.transpose((2, 0, 1))

    if (img > 1).any():
        img = img / 255.0
    return img

def predict_pinbox(full_img, pinbox_color='black0', hybrid=False):

    onnx_dir =  'checkpoints/unet_'+pinbox_color+'.onnx'
    trt_dir =   'checkpoints/unet_'+pinbox_color+'.trt'

    with get_engine(onnx_dir, trt_dir) as engine, engine.create_execution_context() as context:

        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = full_img.astype(np.float32)
        ts= time.time()
        outs_trt = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print("Inference time: ", round(time.time()-ts, 2))
    return outs_trt

def post_process_masks(model_out, type='white',save_masks=False, hybrid=False):

    # Get segmented masks
    if type == 'black0' and hybrid==True:
        size = (3, 361,174)
    elif type =='black0' and hybrid==False:
        size = (3, 352, 139)
    elif type == 'white0' and hybrid==True:
        size = (3, 318, 179)
    elif type == 'white0' and hybrid==False:
        size = (3, 326, 189)
    elif type == 'white1':
        size = (3, 384, 245)
    elif type == 'gray':
        size = (3, 313, 245)

    output = np.array(model_out[0].reshape(size[0],size[1],size[2]), dtype=np.float32)[np.newaxis, ...]
    mask = np.argmax(output, axis=1)
    mask = mask[0].astype(np.int64).squeeze()

    for curr_class in range(3):
        plot = (mask==curr_class)
        hold = np.zeros_like(mask)
        hold[plot] = 255

        if curr_class == 1:
            kernel_erode = np.ones((3, 3), np.uint8)
            kernel_dilate = np.ones((3, 3), np.uint8)
            erode_iter = 2
            dilate_iter = 1
        else:
            kernel_erode = np.ones((3, 3), np.uint8)
            kernel_dilate = np.ones((3, 3), np.uint8)
            erode_iter = 1
            dilate_iter = 1
        hold_ng=hold.astype(np.uint8)
        hold_ng_eroded = cv2.erode(hold_ng, kernel_erode, iterations=erode_iter)
        if curr_class != 1:
            hold_ng_dilated = cv2.dilate(hold_ng_eroded, kernel_dilate, iterations=dilate_iter)
            # print(np.unique(hold_ng_dilated))
            hold = hold_ng_dilated
        elif curr_class == 1:
            hold_ng_dilated = cv2.dilate(hold_ng_eroded, kernel_dilate, iterations=dilate_iter)
            # print(np.unique(hold_ng_dilated))
            hold = hold_ng_dilated
            faults = detect_faults(hold_ng_dilated/255, 'black')
            # print("num faults:", faults)
            # print(np.unique(hold_ng_dilated))
            hold = hold_ng_dilated
        cv2.imwrite('data/out/out_'+str(curr_class)+'.png', hold)

    return faults

def detect_faults(indiv_masks, type):

    not_good = indiv_masks
    bad_edges = cv2.Canny(not_good.astype(np.uint8)*255, 0, 255)
    bad_contours, _ = cv2.findContours(bad_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    bad_contours_clean = []
    for contour in bad_contours:
        if cv2.contourArea(contour) > 10:
            bad_contours_clean.append(contour)
    if len(bad_contours_clean) > 0:
        print("Fault detected in " + type + " pinbox.")
        print("Number of bad "+type+" pins found = " + str(len(bad_contours_clean)))

    return len(bad_contours_clean)

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.8,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0,
                        help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')
    
    return parser.parse_args()


if __name__ == '__main__':

    ts = time.time()
    args = get_args()
    print("\n")
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    in_files = args.input
    black_bbox = [1403, 2590, 1722, 2722]
    # gray = [523, 1909, 810, 2100]
    hybrid=False

    for i, filename in enumerate(in_files):
        logging.info(f'Predicting pins {filename} ...')
        img = Image.open(filename)
        img = np.asarray(img)
        img = img[black_bbox[1]:black_bbox[3], black_bbox[0]:black_bbox[2]]
        # img =  cv2.resize(img , (174, 361))[:,:,::-1]

        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        pre_processed_pin = preprocess_pins(img, pinbox_color='black0', hybrid=hybrid)
        model_out = predict_pinbox(pre_processed_pin, pinbox_color='black0', hybrid=hybrid)
        pin_preds = post_process_masks(model_out, type = 'black0', save_masks=True,hybrid=hybrid)
    print("Output masks saved at: data/out/")
    print("Total runtime: ", str(round((time.time()-ts),2)), 'seconds' + '\n')
