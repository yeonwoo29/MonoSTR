import os
import tqdm
import shutil
import pdb
import torch
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import time


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monodgp'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name
        
        # self.Depth_Metric = Depth_Metric() ### AAAI 2026
        
    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            self.evaluate()

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                self.evaluate()

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)

            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time
            
            # pdb.set_trace()
            # a = self.metric_depth_map(outputs, targets) ## AAAI 2026
            # pdb.set_trace()
            # a = self.Depth_Metric(targets['']) # AAAI 2026   forward(self, depth_logits, gt_boxes2d, num_gt_per_img, gt_center_depth):
            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            results.update(dets)
            progress_bar.update()

        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)

    def save_results(self, results):
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()

    def evaluate(self):
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result

#     # ### AAAI 2026  ######
#     def metric_depth_map(self, outputs, targets):
#         from utils import box_ops
#         depth_map_logits = outputs['pred_depth_map_logits']
#         #pdb.set_trace()
#         #targets = [targets]
#         #targets = {k: v.to('cuda') if isinstance(v, torch.Tensor) else v for k, v in targets.items()}
#         #pdb.set_trace()

#         # num_gt_per_img = [len(t['boxes']) for t in targets]
#         # gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
#         # gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)
#         # gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

#         boxes = targets['boxes'].to('cuda')        # (B, N, 4)
#         depths = targets['depth'].to('cuda')       # (B, N, 1)
#         mask = targets['mask_2d'].to('cuda').bool()  # (B, N)

#         # === 1. 각 이미지당 GT 개수 ===
#         num_gt_per_img = mask.sum(dim=1).tolist()  # length = B

#         # === 2. 마스크로 유효 GT만 추출 ===
#         gt_boxes2d = boxes[mask] * torch.tensor([80, 24, 80, 24], device='cuda')
#         gt_boxes2d = box_ops.box_cxcywh_to_xyxy(gt_boxes2d)  # shape: (sum_valid, 4)

#         gt_center_depth = depths[mask].squeeze(1)  # shape: (sum_valid,)

#         metric_dm = self.Depth_Metric(
#             depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
#         return metric_dm








# ####### AAAI 2026 #####################
# import torch.nn as nn
# class Depth_Metric(nn.Module):

#     def __init__(self):
#         super().__init__()
#         self.device = torch.cuda.current_device()

#     def build_target_depth_from_3dcenter(self, depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img):
#         B, _, H, W = depth_logits.shape
#         depth_maps = torch.zeros((B, H, W), device=depth_logits.device, dtype=depth_logits.dtype)

#         # Set box corners
#         gt_boxes2d[:, :2] = torch.floor(gt_boxes2d[:, :2])
#         gt_boxes2d[:, 2:] = torch.ceil(gt_boxes2d[:, 2:])
#         gt_boxes2d = gt_boxes2d.long()

#         # Set all values within each box to True
#         gt_boxes2d = gt_boxes2d.split(num_gt_per_img, dim=0)
#         gt_center_depth = gt_center_depth.split(num_gt_per_img, dim=0)
#         B = len(gt_boxes2d)
#         for b in range(B):
#             center_depth_per_batch = gt_center_depth[b]
#             center_depth_per_batch, sorted_idx = torch.sort(center_depth_per_batch, dim=0, descending=True)
#             gt_boxes_per_batch = gt_boxes2d[b][sorted_idx]
#             for n in range(gt_boxes_per_batch.shape[0]):
#                 u1, v1, u2, v2 = gt_boxes_per_batch[n]
#                 depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]

#         return depth_maps
    
#     def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
#         bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
#         indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    
#         if target:
#             # Remove indicies outside of bounds
#             mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
#             indices[mask] = num_bins

#             # Convert to integer
#             indices = indices.type(torch.int64)
       
#         return indices

#     def forward(self, depth_logits, gt_boxes2d, num_gt_per_img, gt_center_depth):
#         # Bin depth map to create target
#         depth_maps = self.build_target_depth_from_3dcenter(depth_logits, gt_boxes2d, gt_center_depth, num_gt_per_img)
        
#         return loss