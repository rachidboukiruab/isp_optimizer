from skopt import gp_minimize
from thirdparty.openISP.pipeline import Pipeline
from detector.detector import Detector
import time
import skimage.io
import numpy as np


class Optimizer:

    def __init__(self, isp_conf, optimizer_conf, dataset_dict):

        self.isp_conf = isp_conf
        self.optimizer_conf = optimizer_conf
        self.dataset_dict = dataset_dict
        self.yolo_model = Detector()
        self.gamma_bounds = []
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                self.gamma_bounds.append(self.optimizer_conf["bounds"][module][param])

        self.init_values = []
        for module in self.optimizer_conf["init"]:
            for param in self.optimizer_conf["init"][module]:
                self.init_values.append(self.optimizer_conf["init"][module][param])

    def cost_function(self, x):
        text = ""
        i = 0
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                with self.isp_conf.unfreeze():
                    self.isp_conf[module][param] = x[i]
                text += '{}: {:.2f} \t'.format(param, x[i])
                i += 1

        iou, mAP, time = self.batch_image_processing()

        print(f'{text} \t mAP: {mAP*100:.3f}% \t Computation Time (s): {time:.5f}')
        return 1 - mAP

    def bayesian_optimization(self, acq_func="EI", acq_optimizer="sampling", verbose=True):
        result = gp_minimize(self.cost_function, self.gamma_bounds, acq_func=acq_func, acq_optimizer=acq_optimizer,
                             xi=self.init_values, verbose=verbose)
        return result

    def batch_image_processing(self):
        n = 0
        sum_iou = 0
        sum_mAP = 0
        start_time = time.time()
        for img_name in self.dataset_dict.keys():
            # Process image
            pipeline = Pipeline(self.isp_conf)
            bayer = skimage.io.imread(self.dataset_dict[img_name]['img_path']).astype(np.uint8)
            data, _ = pipeline.execute(bayer, verbose=False)
            processed_img = data['output']

            # Perform image detection and calculate IoU and mAP
            iou, mAP = self.yolo_model.inference(processed_img, visualize_img=False,
                                                 gt_bboxes=self.dataset_dict[img_name]['bboxes'],
                                                 classes=["car", "truck"], min_conf=0.5, show_gt=True,
                                                 save_img=False, out_path="")
            n += 1
            sum_iou += iou
            sum_mAP += mAP

        # Calculate global mAP
        mean_iou = sum_iou / n
        mean_mAP = sum_mAP / n
        end_time = time.time() - start_time
        #print(f'IoU: {mean_iou:.3f} \t mAP: {mean_mAP:.3f} \t Computation Time (s): {end_time:.5f}')
        return mean_iou, mean_mAP, end_time