from thirdparty.openISP.utils.yacs import Config
from optimizer.optimizer import Optimizer
import os


if __name__ == "__main__":
    # Load the configuration file 
    cfg = Config('configs\main_config.yaml')
    
    # In case the out folder does not exist, create it.
    # In this folder we will save the images after inference in case the save_img flag is true and
    # the optimization results.
    out_folder = cfg["detector"]["output_folder"]
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    # Initialize the optimizer class with the configuration file
    opt = Optimizer(cfg)

    # Check which optimizer is selected and run its corresponding function
    # In case we put "inference" as optimizer type, we will evaluate the validation dataset and print the mAP and IoU result.
    if cfg['optimizer']['optimizer_type'] == 'bayesian':
        result, mean_metrics_dict = opt.bayesian_optimization(acq_func="EI", acq_optimizer="sampling", verbose=True)
        yaml_file = Config(mean_metrics_dict)
        yaml_file.dump('out/optimization_data_bayesian.yaml')
        print(result)
    elif cfg['optimizer']['optimizer_type'] == 'cma':
        result, mean_metrics_dict = opt.cma_optimization()
        yaml_file = Config(mean_metrics_dict)
        yaml_file.dump('out/optimization_data_cma.yaml')
        print(result)
    elif cfg['optimizer']['optimizer_type'] == 'inference':
        mean_iou, mean_mAP, end_time = opt.batch_image_processing()
        print(f'IoU: {mean_iou:.3f} \t mAP: {mean_mAP:.3f} \t Computation Time (s): {end_time:.5f}')
    else:
        print("No valid optimizer type.")










