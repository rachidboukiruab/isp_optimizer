from skopt import gp_minimize
from thirdparty.openISP.pipeline import Pipeline
from detector.detector import Detector
from thirdparty.openISP.utils.yacs import Config
import time
import skimage.io
import numpy as np
import cma


class Optimizer:

    def __init__(self, cfg, dataset_dict):
        self.isp_conf = Config(cfg["isp"]["config_file"])
        self.optimizer_conf = Config(cfg["optimizer"])
        self.detector_conf = Config(cfg["detector"])
        self.norm = self.optimizer_conf["normalization"]
        self.dataset_dict = dataset_dict
        self.yolo_model = Detector()
        self.bounds = []
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                self.bounds.append(self.optimizer_conf["bounds"][module][param])

        self.init_values = []
        for module in self.optimizer_conf["init"]:
            for param in self.optimizer_conf["init"][module]:
                self.init_values.append(self.optimizer_conf["init"][module][param])

        self.init_values_norm = [(self.init_values[i]-self.bounds[i][0])/(self.bounds[i][1]-self.bounds[i][0])
                                 for i in range(len(self.init_values))]
        self.denorm_constants = [[self.bounds[i][1]-self.bounds[i][0], self.bounds[i][0]]
                                 for i in range(len(self.init_values))]


    def cost_function(self, x):
        text = ""
        i = 0
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                with self.isp_conf.unfreeze():
                    if self.norm:
                        self.isp_conf[module][param] = x[i]*self.denorm_constants[i][0] + self.denorm_constants[i][1]
                    else:
                        self.isp_conf[module][param] = x[i]
                #text += '{}: {:.2f} \t'.format(param, x[i])
                text += '{}: {:.2f} \t'.format(param, self.isp_conf[module][param])
                i += 1

        iou, mAP, time = self.batch_image_processing()

        print(f'{text} \tmAP: {mAP*100:.3f}% \tComputation Time (s): {time:.5f}')
        return 1 - mAP

    def batch_image_processing(self):
        n = 0
        sum_iou = 0
        sum_mAP = 0
        start_time = time.time()
        for img_name in self.dataset_dict.keys():
            # Process image
            pipeline = Pipeline(self.isp_conf)
            bayer = skimage.io.imread(self.dataset_dict[img_name]['img_path']).astype(np.uint8)
            data, _ = pipeline.execute(bayer, verbose=self.detector_conf["verbose"])
            processed_img = data['output']

            # Perform image detection and calculate IoU and mAP
            iou, mAP = self.yolo_model.inference(processed_img, visualize_img=self.detector_conf["show_img"],
                                                 gt_bboxes=self.dataset_dict[img_name]['bboxes'],
                                                 classes=["car", "truck"], min_conf=0.5, show_gt=True,
                                                 save_img=self.detector_conf["save_img"], out_path=self.detector_conf["output_folder"])
            n += 1
            sum_iou += iou
            sum_mAP += mAP

        # Calculate global mAP
        mean_iou = sum_iou / n
        mean_mAP = sum_mAP / n
        end_time = time.time() - start_time
        #print(f'IoU: {mean_iou:.3f} \t mAP: {mean_mAP:.3f} \t Computation Time (s): {end_time:.5f}')
        return mean_iou, mean_mAP, end_time

    def bayesian_optimization(self, acq_func="EI", acq_optimizer="sampling", verbose=True):
        if self.norm:
            boundaries = [[0., 1.] for i in range(len(self.bounds))]
            x0 = self.init_values_norm
        else:
            boundaries = self.bounds
            x0 = self.init_values

        result = gp_minimize(self.cost_function, boundaries, acq_func=acq_func, acq_optimizer=acq_optimizer,
                             xi=x0, verbose=verbose)
        return result

    def cma_optimization(self):
        if self.norm:
            lower_bounds = 0.
            upper_bounds = 1.
            x0 = self.init_values_norm
        else:
            lower_bounds = [self.bounds[i][0] for i in range(len(self.bounds))]
            upper_bounds = [self.bounds[i][1] for i in range(len(self.bounds))]
            x0 = self.init_values

        xopt, es = cma.fmin2(self.cost_function, x0, 1, {'bounds': [lower_bounds, upper_bounds]})
        return xopt