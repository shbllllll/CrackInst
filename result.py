from torch import device
import torch
import os
from mmdet.apis import init_detector, show_result_pyplot, inference_detector

#config_file = '/data/ss/crackSeg/crackseg/configs/solov2/solov2_r50_fpn_1x_coco.py'
# config_file = '/data/ss/crackSeg/crackseg/configs/solov2/solov2_light_v13.py'
config_file='/data/ss/crackseg/work_dirs/point_rend_r50_caffe_fpn_mstrain_1x_coco/point_rend_r50_caffe_fpn_mstrain_1x_coco.py'

# checkpoint_file = '/data/ss/crackSeg/crackseg/work_dirs/solov2_light_v13/trash.pth'
checkpoint_file='/data/ss/crackseg/work_dirs/point_rend_r50_caffe_fpn_mstrain_1x_coco/epoch_200.pth'
#checkpoint_file = '/data/ss/crackSeg/crackseg/work_dirs/solov2_r50_fpn_1x_coco/epoch_200.pth'


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = init_detector(config_file, checkpoint_file, device=device)

#img_file = '/data/ss/crackSeg/crackseg/data/trash/train/vid_000034_frame0000008.jpg'
img_file = '/data/ss/crackseg/data/trash/train/vid_000440_frame0000103.jpg'
# vid_000083_frame0000008   vid_000290_frame0000018   vid_000440_frame0000103   vid_000550_frame0000028

savepath = 'runs/pointRend'
# savepath = 'runs/mask/'
# savepath = 'runs/point rend'

result = inference_detector(model, img_file)
print(result)

out_file = os.path.join(savepath, 'vid_000440_frame0000103.jpg')

print(out_file)

show_result_pyplot(model, img_file, result, score_thr=0.8, out_file=out_file)

# # 模型配置文件-
#     config_file = 'configs/A_proposed/our.py'
# # 预训练模型文件,
#     checkpoint_file = '/home/hws/mmdetection-master/work_dirs/our/latest.pth'
# # 通过模型配置文件与预训练文件构建模型
#     model = init_detector(config_file, checkpoint_file)
# # 图片路径
#     img_dir = '/home/hws/mmdetection-master/data/coco/jpg'
#     # 检测后存放图片路径
#     out_dir = 'result/'
#     if not os.path.exists(out_dir):
#         os.mkdir(out_dir)
#
#     # 测试集的图片名称txt
#     test_path = '/home/hws/mmdetection-master/data/coco/jpg/000002.jpg'

