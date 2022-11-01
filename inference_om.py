import cv2
import os
from matplotlib import image
# os.system('. ~/mindx_dir/mxVision/set_env.sh')
import numpy as np
# import mindx.sdk as sdk
from mindx.sdk.base import Tensor, Model
import torch.nn.functional as F
import torch


def infer(filepath, ROOT_PATH, SAVE_PATH, device_id):
    # np.array(
    model = Model(filepath, device_id)
    print(model)
    os.makedirs(SAVE_PATH, exist_ok=True)
    for img_name in os.listdir(ROOT_PATH):
        img_src = cv2.imread(os.path.join(ROOT_PATH, img_name))
        resizeImage = cv2.resize(img_src, (352,352), interpolation= cv2.INTER_LINEAR)
        # resizeImage = cv2.cvtColor(resizeImage, cv2.COLOR_BGR2RGB)
        img_in = np.transpose(resizeImage, (2, 0, 1)).astype(np.float32)
        img_in = np.expand_dims(img_in, axis=0)
        img_in /= 255.0
        
        # resizeImage = resizeImage.transpose((2, 0, 1))
        # r_img_ary = np.array(resizeImage)
        # r_img_ary = r_img_ary.astype(np.float32)
        # r_img_ary = np.expand_dims(r_img_ary, axis=0)
        # r_img_ary /= 255.0
        
        imageTensor = Tensor(img_in)
        imageTensor.to_device(device_id)

        out = model.infer(imageTensor)
        out = out[0]
        out.to_host()

        out = torch.from_numpy(np.array(out))
        res = F.upsample(out, size=img_src.shape[:2], mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # out_npy = np.resize(out_npy, img_src.shape[:2])
        # res=1/(1+(np.exp((-out_npy))))
        # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # print('=>out_npy', out_npy.shape, out_npy.max(), out_npy.min())
        res = 255.0 * res
        # print('=>res2', res.shape, res.max(), res.min())
        res = res.astype(np.uint8)
        # print('=>res3', res)
        cv2.imwrite(os.path.join(SAVE_PATH, img_name), res)



if __name__ == "__main__":
    
    infer(
        filepath='./Net_epoch_best_sim.om.om', 
        ROOT_PATH='/home/fandengping01/gepeng_project/DGNet-Pytroch/data/NC4K/Imgs', 
        SAVE_PATH='seg_results_om',
        device_id=0)