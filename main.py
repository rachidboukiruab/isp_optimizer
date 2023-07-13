from thirdparty.openISP.utils.yacs import Config
from dataset.dataset import Dataset
from optimizer.optimizer import Optimizer


if __name__ == "__main__":
    cfg = Config('configs\main_config_vocDataset.yaml')
    imgs_folder = cfg['train_folder']

    dataset = Dataset()
    if cfg['dataset_type'] == 'voc':
        dataset_dict = dataset.VOC_dataset(imgs_folder)
    else:
        csv_path = cfg['annotations_file']
        dataset_dict = dataset.RAW_cars_dataset(csv_path, imgs_folder)
    # dataset_dict = {'bboxes': [[xmin, ymin, xmax, ymax]], 'img_path': ''}

    opt = Optimizer(cfg, dataset_dict)

    if cfg['optimizer']['optimizer_type'] == 'bayesian':
        result = opt.bayesian_optimization(acq_func="EI", acq_optimizer="sampling", verbose=True)
        print(result)
    else:
        if cfg['optimizer']['optimizer_type'] == 'cma':
            result = opt.cma_optimization()
            print(result)










