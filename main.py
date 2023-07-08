from thirdparty.openISP.utils.yacs import Config
from dataset.dataset import Dataset
from optimizer.optimizer import Optimizer


if __name__ == "__main__":
    cfg = Config('configs\main_config.yaml')
    isp_conf = Config(cfg["isp"]["config_file"])
    optimizer_conf = Config(cfg["optimizer"])

    out_folder = cfg["detector"]["input_folder"]
    csv_path = cfg['annotations_file']
    imgs_folder = cfg['val_folder']

    dataset = Dataset()
    #dataset_dict = dataset.VOC_dataset(imgs_folder)
    dataset_dict = dataset.RAW_cars_dataset(csv_path, imgs_folder) # {'bboxes': [[xmin, ymin, xmax, ymax]], 'img_path': ''}

    opt = Optimizer(isp_conf, optimizer_conf, dataset_dict)

    result = opt.bayesian_optimization(acq_func="EI", acq_optimizer="sampling", verbose=True)

    print(result)







