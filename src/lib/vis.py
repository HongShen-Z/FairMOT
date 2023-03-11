import torch

from models.decode import mot_decode
from models.model import create_model, load_model
from models.utils import _tranpose_and_gather_feat
from tracking_utils.utils import *
from utils.post_process import ctdet_post_process
import cv2
from opts import opts


def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width]
    ratio = min(float(height) / shape[0], float(width) / shape[1])
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh


def load_img(img_path):
    # Read image
    img0 = cv2.imread(img_path)  # BGR
    assert img0 is not None, 'Failed to load ' + img_path

    # Padded resize
    img, _, _, _ = letterbox(img0)

    # Normalize RGB
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img, dtype=np.float32)
    img /= 255.0

    # cv2.imwrite(img_path + '.letterbox.jpg', 255 * img.transpose((1, 2, 0))[:, :, ::-1])  # save letterbox image
    return img, img0


class JDETracker(object):
    def __init__(self, opt, frame_rate=30):
        self.opt = opt
        if opt.gpus[0] >= 0:
            opt.device = torch.device('cuda')
        else:
            opt.device = torch.device('cpu')
        print('Creating model...')
        self.model = create_model(opt.arch, opt.heads, opt.head_conv)
        self.model = load_model(self.model, opt.load_model)
        # self.model = self.model.to(opt.device)
        self.model.eval()

        self.frame_id = 0
        self.det_thresh = opt.conf_thres
        self.buffer_size = int(frame_rate / 30.0 * opt.track_buffer)
        self.max_time_lost = self.buffer_size
        self.max_per_image = opt.K

    def post_process(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(1, -1, dets.shape[2])
        dets = ctdet_post_process(
            dets.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], self.opt.num_classes)
        for j in range(1, self.opt.num_classes + 1):
            dets[0][j] = np.array(dets[0][j], dtype=np.float32).reshape(-1, 5)
        return dets[0]

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.opt.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)

        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.opt.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.opt.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def update(self, im_blob, img0):
        width = img0.shape[1]
        height = img0.shape[0]
        inp_height = im_blob.shape[2]
        inp_width = im_blob.shape[3]
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(float(inp_width) / float(inp_height) * height, width) * 1.0
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.opt.down_ratio,
                'out_width': inp_width // self.opt.down_ratio}

        ''' Step 1: Network forward, get detections & embeddings'''
        with torch.no_grad():
            im_blob = torch.from_numpy(im_blob)
            output = self.model(im_blob)
            hm = output['hm'].sigmoid_()
            wh = output['wh']
            id_feature = output['id']
            id_feature = F.normalize(id_feature, dim=1)

            reg = output['reg'] if self.opt.reg_offset else None
            dets, inds = mot_decode(hm, wh, reg=reg, ltrb=self.opt.ltrb, K=self.opt.K)
            id_feature = _tranpose_and_gather_feat(id_feature, inds)
            id_feature = id_feature.squeeze(0)
            id_feature = id_feature.cpu().numpy()

        dets = self.post_process(dets, meta)
        dets = self.merge_outputs([dets])[1]

        remain_inds = dets[:, 4] > self.opt.conf_thres
        dets = dets[remain_inds]
        id_feature = id_feature[remain_inds]

        # vis
        print(torch.max(hm))
        print(len(dets))
        # 读取原始图片
        orig_img = cv2.imread('../000045.jpg')

        feature_map = torch.squeeze(hm)
        feature_map = np.asarray(feature_map)

        # 将特征图数据归一化到0-255范围，并转换为整数类型
        feature_map = (feature_map * 255).astype(np.uint8)
        # 使用cv2.applyColorMap函数将特征图转换为热力图，选择COLORMAP_JET作为颜色映射
        heatmap = cv2.applyColorMap(feature_map, cv2.COLORMAP_JET)
        # 将热力图调整到和原始图片一样的大小
        heatmap = cv2.resize(heatmap, (orig_img.shape[1], orig_img.shape[0]))
        # 将热力图和原始图片按一定比例叠加，得到最终的图片
        img_heatmap = cv2.addWeighted(orig_img, 0.6, heatmap, 0.4, 0)
        # 保存图片为heatmap.jpg文件
        cv2.imwrite('../heatmap.jpg', img_heatmap)

        print(hm.shape)


if __name__ == '__main__':
    opt = opts().init()
    img_path = '../000045.jpg'
    img, img0 = load_img(img_path)
    img = np.expand_dims(img, 0)
    tracker = JDETracker(opt)
    tracker.update(img, img0)
