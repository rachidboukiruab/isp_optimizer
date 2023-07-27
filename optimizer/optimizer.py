from skopt import gp_minimize
from thirdparty.openISP.pipeline import Pipeline
from detector.detector import Detector
from thirdparty.openISP.utils.yacs import Config
import time
import skimage.io
import numpy as np
import cma
import concurrent.futures
import threading


class Optimizer:

    def __init__(self, cfg, dataset_dict):
        self.isp_conf = Config(cfg["isp"]["config_file"])
        self.multithread = cfg["isp"]["multithread"]
        self.optimizer_conf = Config(cfg["optimizer"])
        self.detector_conf = Config(cfg["detector"])
        self.norm = self.optimizer_conf["normalization"]
        self.dataset_dict = dataset_dict
        self.yolo_model = Detector(ficosa_model=self.detector_conf["ficosa_model"],
                                   model_path=self.detector_conf["model_path"])
        self.classes = list(self.detector_conf["classes"])

        self.bounds = []
        self.init_values = []
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                if type(self.optimizer_conf["bounds"][module][param][0]) == tuple:
                    for value in self.optimizer_conf["bounds"][module][param]:
                        self.bounds.append(value)
                    for value in self.optimizer_conf["init"][module][param]:
                        self.init_values.append(value)
                else:
                    self.bounds.append(self.optimizer_conf["bounds"][module][param])
                    self.init_values.append(self.optimizer_conf["init"][module][param])

        self.init_values_norm = [(self.init_values[i]-self.bounds[i][0])/(self.bounds[i][1]-self.bounds[i][0])
                                 for i in range(len(self.init_values))]
        self.denorm_constants = [[self.bounds[i][1]-self.bounds[i][0], self.bounds[i][0]]
                                 for i in range(len(self.init_values))]

        self.lock = threading.Lock()

    def relaxed_hyperparameter(self, bounds, hyp):
        lowest_value = bounds[0]
        highest_value = bounds[1]
        B = highest_value - lowest_value
        relaxed_hyp = (hyp - lowest_value)/B-1
        return relaxed_hyp

    def cost_function(self, x):
        text = ""
        i = 0
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                if type(self.optimizer_conf["bounds"][module][param][0]) == tuple:
                    isp_param = []
                    str = '[ '
                    for value in self.optimizer_conf["bounds"][module][param]:
                        k = int(x[i]*self.denorm_constants[i][0] + self.denorm_constants[i][1])
                        isp_param.append(k)
                        i +=1
                        str += '{} '.format(k)
                    with self.isp_conf.unfreeze():
                        self.isp_conf[module][param] = tuple(isp_param)
                    text += '{}: {}] \t'.format(param, str)
                else:
                    with self.isp_conf.unfreeze():
                        if self.norm:
                            self.isp_conf[module][param] = x[i]*self.denorm_constants[i][0] + self.denorm_constants[i][1]
                        else:
                            self.isp_conf[module][param] = x[i]
                    #text += '{}: {:.2f} \t'.format(param, x[i])
                    i += 1
                    text += '{}: {:.2f} \t'.format(param, self.isp_conf[module][param])

        iou, mAP, time = self.batch_image_processing()

        print(f'{text} \tmAP: {mAP*100:.3f}% \tComputation Time (s): {time:.5f}')
        return 1 - mAP

    def process_image(self, img_name):
        pipeline = Pipeline(self.isp_conf)
        bayer = skimage.io.imread(self.dataset_dict[img_name]['img_path']).astype(np.uint8)
        data, _ = pipeline.execute(bayer, verbose=self.detector_conf["verbose"])
        processed_img = data['output']
        img_out_path = self.detector_conf["output_folder"] + img_name + ".jpg"
        # Lock de access to the shared class
        self.lock.acquire()
        try:
            # Perform image detection and calculate IoU and mAP
            iou, mAP = self.yolo_model.inference(processed_img, visualize_img=self.detector_conf["show_img"],
                                                 gt_bboxes=self.dataset_dict[img_name]['bboxes'],
                                                 classes=self.classes, min_conf=0.5,
                                                 show_gt=self.detector_conf["show_gt"],
                                                 save_img=self.detector_conf["save_img"], out_path=img_out_path)
        finally:
            # Lock release after finishing the use of shared class
            self.lock.release()
        return iou, mAP

    def batch_image_processing(self):
        n = 0
        sum_iou = 0
        sum_mAP = 0
        start_time = time.time()
        # Process image
        if self.multithread:
            # Create a ThreadPoolExecutor with the desired number of threads (adjust as needed)
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_img_name = {executor.submit(self.process_image, img_name): img_name
                                      for img_name in self.dataset_dict.keys()}

                for future in concurrent.futures.as_completed(future_to_img_name):
                    img_name = future_to_img_name[future]
                    try:
                        iou, mAP = future.result()
                        n += 1
                        sum_iou += iou
                        sum_mAP += mAP
                    except Exception as e:
                        print(f"Error processing image {img_name}: {e}")
        else:
            for img_name in self.dataset_dict.keys():
                iou, mAP = self.process_image(img_name)

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
        x_denorm = self.convert_results(xopt)
        return x_denorm

    def convert_results(self, x):
        x_denorm = []
        i = 0
        for module in self.optimizer_conf["bounds"]:
            for param in self.optimizer_conf["bounds"][module]:
                value = 0
                if self.norm:
                    value = x[i] * self.denorm_constants[i][0] + self.denorm_constants[i][1]
                else:
                    value = x[i]
                x_denorm.append(value)
                i += 1

        return x_denorm
