from skopt import gp_minimize
from detector.detector import Detector
from dataset.dataset import CustomVOC, ImageProcessingTransform
from thirdparty.openISP.utils.yacs import Config
import time
import skimage.io
import numpy as np
import cma
from torchvision import transforms
from torch.utils.data import DataLoader
from optimizer.utils import norm_hyperparameter, collate_fn


class Optimizer:
    """ Core ISP optimizer """
    def __init__(self, cfg):
        """
        :param cfg: yacs.Config object, configurations about dataset, optimizer, and detector modules.
        """
        # Load the configuration for each submodule
        self.dataset_conf = Config(cfg["dataset"])
        self.isp_conf = Config(cfg["isp"]["config_file"])
        self.optimizer_conf = Config(cfg["optimizer"])
        self.detector_conf = Config(cfg["detector"])
        self.norm = self.optimizer_conf["normalization"]
        self.classes = list(self.detector_conf["classes"])

        # If the optimizer type is inference, select the validation dataset instead of training set
        if self.optimizer_conf['optimizer_type']=='inference':
            self.dataset = CustomVOC(self.dataset_conf['val_folder'], self.dataset_conf['classes'],
                                     transform=ImageProcessingTransform(self.isp_conf, verbose=self.detector_conf["verbose"]))
        else:
            self.dataset = CustomVOC(self.dataset_conf['train_folder'], self.dataset_conf['classes'],
                                     transform=ImageProcessingTransform(self.isp_conf, verbose=self.detector_conf["verbose"]))

        # Initialize the detector submodule.
        self.yolo_model = Detector(ficosa_model=self.detector_conf["ficosa_model"],
                                   model_path=self.detector_conf["model_path"])

        # Load the upper and lower bounds of each parameter we want to optimize in the ISP
        # Also load the initial values of that hyperparameters
        self.bounds = []
        self.init_values = []
        # To not repeat again the same loop, we use the same for loop to load the hyperparameters keys in
        # dict where we will save the evaluation metrics for each optimization loop.
        self.mean_metrics_dict = {'Hyperparameters': {}} 
        for module in self.optimizer_conf["bounds"]:
            self.mean_metrics_dict['Hyperparameters'][module] = {}
            for param in self.optimizer_conf["bounds"][module]:
                self.mean_metrics_dict['Hyperparameters'][module][param] = []
                if type(self.optimizer_conf["bounds"][module][param][0]) == tuple:
                    for value in self.optimizer_conf["bounds"][module][param]:
                        self.bounds.append(value)
                    for value in self.optimizer_conf["init"][module][param]:
                        self.init_values.append(value)
                else:
                    self.bounds.append(self.optimizer_conf["bounds"][module][param])
                    self.init_values.append(self.optimizer_conf["init"][module][param])

        # If normalization flag is active. Normalize the hyperparameters and calculate the denormalization constants
        if self.norm:
            self.init_values_norm = [norm_hyperparameter(self.bounds[i], self.init_values[i])
                                    for i in range(len(self.init_values))]
            self.denorm_constants = [[self.bounds[i][1]-self.bounds[i][0], self.bounds[i][0]]
                                    for i in range(len(self.init_values))]

        # Create a key in mean_metrics_dict for each class defined
        for cls in self.classes:
            self.mean_metrics_dict[cls] = {'mAP': [], 'IoU': [], 'Precision': [], 'Recall': []}

    def cost_function(self, x):
        """
        Definition of the cost function that will be used in the optimizers. This function loads the 
        proposed X values for the ISP hyperparameters, processes the batch of the training images with the
        new proposed hyperparameters, performs the inference and extracts the mAP of the detections and returns 
        1-mAP as the loss we want to minimize.
        x: list of the proposed values for the ISP hyperparameters
        :return: The loss function that we want to minimize (1 - mAP)
        """
        text = ""
        i = 0
        # Load the hyperparameters proposed by the optimizer in the ISP config file
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
                    self.mean_metrics_dict['Hyperparameters'][module][param].append(self.isp_conf[module][param])
                else:
                    with self.isp_conf.unfreeze():
                        if self.norm:
                            self.isp_conf[module][param] = x[i]*self.denorm_constants[i][0] + self.denorm_constants[i][1]
                        else:
                            self.isp_conf[module][param] = x[i]
                    #text += '{}: {:.2f} \t'.format(param, x[i])
                    i += 1
                    text += '{}: {:.2f} \t'.format(param, self.isp_conf[module][param])
                    self.mean_metrics_dict['Hyperparameters'][module][param].append(float(self.isp_conf[module][param]))
        # Call the function that processes the batch of images, performs the inference and extract the iou and mAP metrics
        iou, mAP, time = self.batch_image_processing()

        print(f'{text} \tmAP: {mAP*100:.3f}% \tComputation Time (s): {time:.5f}')
        return 1 - mAP

    def batch_image_processing(self):
        """
        Processes the batch of images with the new ISP configuration and injects them to the detector submodule from where
        the inference and evaluation is performed.
        :return: The IoU, mAP and computation time in seconds
        """
        start_time = time.time()
        print("Starting ISP processing...")
        begin = time.time_ns()
        self.dataset.update_transform(transform=ImageProcessingTransform(self.isp_conf, verbose=self.detector_conf["verbose"]))
        data_loader = DataLoader(self.dataset, batch_size=self.dataset_conf['batch_size'],
                                 shuffle=self.dataset_conf['shuffle'],collate_fn=collate_fn,
                                 num_workers=self.dataset_conf['num_workers'])
        n = 0
        map = 0.
        mean_iou = 0.
        for images, bboxes in data_loader:
            print(f"ISP processing time: {(time.time_ns() - begin) / 1000000} ms")
            img_out_path = self.detector_conf["output_folder"]
            # Perform image detection and calculate IoU and mAP
            iter_metrics_dict = self.yolo_model.inference(images, bboxes,
                                                          visualize_img=self.detector_conf["show_img"],
                                                          classes=self.classes, min_conf=0.5,
                                                          show_gt=self.detector_conf["show_gt"],
                                                          save_img=self.detector_conf["save_img"], out_path=img_out_path)
            for cls in self.classes:
                if cls in iter_metrics_dict.keys():
                    iou = float(np.mean(iter_metrics_dict[cls]['IoU']))
                    ap = float(np.mean(iter_metrics_dict[cls]['AP']))
                    self.mean_metrics_dict[cls]['IoU'].append(iou)
                    self.mean_metrics_dict[cls]['mAP'].append(ap)
                    self.mean_metrics_dict[cls]['Precision'].append(float(np.mean(iter_metrics_dict[cls]['Precision'])))
                    self.mean_metrics_dict[cls]['Recall'].append(float(np.mean(iter_metrics_dict[cls]['Recall'])))
                    map += ap
                    mean_iou += iou
                    n+=1
                else:
                    self.mean_metrics_dict[cls]['IoU'].append(0.)
                    self.mean_metrics_dict[cls]['mAP'].append(0.)
                    self.mean_metrics_dict[cls]['Precision'].append(0.)
                    self.mean_metrics_dict[cls]['Recall'].append(0.)
            break
        end_time = time.time() - start_time
        if n > 0:
            map = map/float(n)
            mean_iou = mean_iou/float(n)
        else:
            map = 0.
            mean_iou = 0.

        #print(f'IoU: {mean_iou:.3f} \t mAP: {mean_mAP:.3f} \t Computation Time (s): {end_time:.5f}')
        if self.optimizer_conf['optimizer_type'] == 'inference':
            print(self.mean_metrics_dict)
        return mean_iou, map, end_time

    def bayesian_optimization(self, acq_func="EI", acq_optimizer="sampling", verbose=True):
        """
        Optimizes the ISP hyperparameters using Bayesian algorithm from skopt.gp_minimize object.
        :param acq_func: String. Default value "EI"
        :param acq_optimizer: String. Default value "sampling".
        :param verbose: Boolean. Default value True.
        :return: Tuple of a list of the ISP hyperparameters that minimizes the defined cost function and a dictionary
                of the detail of hyperparameters used in each iteration and the result in IoU, mAP, Precision, Recall.
        """
        try:
            if self.norm:
                boundaries = [[0., 1.] for i in range(len(self.bounds))]
                x0 = self.init_values_norm
            else:
                boundaries = self.bounds
                x0 = self.init_values

            result = gp_minimize(self.cost_function, boundaries, acq_func=acq_func, acq_optimizer=acq_optimizer,
                                 xi=x0, verbose=verbose)
            return result, self.mean_metrics_dict
        except KeyboardInterrupt:
            return None, self.mean_metrics_dict

    def cma_optimization(self):
        """
        Optimizes the ISP hyperparameters using Covariance Matrix Adaptation (CMA) algorithm from cma.fmin2 object.
        :return: Tuple of a list of the ISP hyperparameters that minimizes the defined cost function and a dictionary
                of the detail of hyperparameters used in each iteration and the result in IoU, mAP, Precision, Recall.
        """
        try:
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
            return x_denorm, self.mean_metrics_dict
        except KeyboardInterrupt:
            return None, self.mean_metrics_dict

    def convert_results(self, x):
        """
        Denormalizes a given list of ISP hyperparameters.
        :param x: List of ISP hyperparameters
        :return: List of denormalized ISP hyperparameters.
        """
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
