import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
from PIL import Image
from lib import VideoModel_pvtv2 as Network
from dataloaders import test_dataloader
import imageio
import pdb
import sys
from tqdm import tqdm
from mindx.sdk.base import Tensor, Model
# os.system('. ~/mindx_dir/mxVision/set_env.sh')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str, default='MoCA')
parser.add_argument('--testsplit',  type=str, default='TestDataset_per_sq')
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--trainsize', type=int, default=352, help='testing size')
parser.add_argument('--pth_path', type=str, default='/Users/mac/Downloads/snapshot/Net_epoch_MoCA_short_term_pseudo.pth')
parser.add_argument('--pretrained_cod10k', default=None,
                        help='path to the pretrained Resnet')
opt = parser.parse_args()

def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "aux" not in name)/1e6

if __name__ == '__main__':
    test_loader = test_dataloader(opt)
    save_root = '/home/fandengping01/shuowang_project/sltnet_om_res/{}/'.format(opt.dataset)
    # pdb.set_trace()
    # model = Network(opt)

    device_id = 0

    model = Model('/home/fandengping01/shuowang_project/test/sltnet_sim_om.om', device_id)

    # pretrained_dict = torch.load(opt.pth_path)
    # model_dict = model.state_dict()
    # #pdb.set_trace()
    # for k, v in pretrained_dict.items():
    #     pdb.set_trace()

    # model.load_state_dict(torch.load(opt.pth_path, map_location=torch.device('cpu')))
    # model.cuda()
    # model.eval()

    # onnx_fp = '/Users/mac/Downloads/sltnet.onnx'
    # input_names = ["image"]  
    # output_names = ["pred"]  
    # dynamic_axes = {'image': {0: '-1'}, 'pred': {0: '-1'}} 
    # dummy_input = torch.randn(1, 9, 352, 352).numpy()
    # torch.onnx.export(model, dummy_input, onnx_fp, input_names=input_names, dynamic_axes=dynamic_axes, output_names=output_names, opset_version=11, verbose=True) 

    # imageTensor = Tensor(dummy_input)
    # imageTensor.to_device(device_id)
    # out = model.infer(imageTensor)
    # out = out[0]
    # out.to_host()

    # compute parameters
    # print('Total Params = %.2fMB' % count_parameters_in_MB(model))

    # print('debug')

    # out = torch.from_numpy(np.array(out))
    # print(out.shape)

    # sys.exit()

    for i in tqdm(range(test_loader.size)):
        images, gt, names, scene = test_loader.load_data()
        save_path=save_root+scene+'/Pred/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        # images = [x.cuda() for x in images]

        images = torch.cat(images, dim=1)


        images = images.numpy()
        imageTensor = Tensor(images)
        imageTensor.to_device(device_id)
        out = model.infer(imageTensor)
        res = out[-1]
        res.to_host()

        # res1, res2, res = model(images)

        res = torch.from_numpy(np.array(res))

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # res = F.upsample(res1[-1], size=gt.shape, mode='bilinear', align_corners=False)
        # res = F.upsample(res2[-1], size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        print('> ')

        name =names[0].replace('jpg','png')

        # if name[-5] in ['0','5']:
        fp = save_path+name
        imageio.imwrite(save_path+name, res)

        print(fp)
