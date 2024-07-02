import os
from pylabel import importer

def load_dataset(annotations_folder_path, classes_list = ["car", "person", "motorbike", "truck"]):
    """
    Load dataset and preprocess it into a dictionary format.

    Args:
        annotations_folder_path (str): Path to the folder containing the annotation files.
        classes_list (List[str]): List of classes to be considered in the dataset. Defaults to ["car", "person", "motorbike", "truck"].

    Returns:
        dict: A dictionary where each key is an image ID and each value is a dictionary containing the image path and bounding boxes.
        Example:
        {
            '000': {
                'img_path': '/home/rachid/raw_dataset/test/000.raw',
                'bboxes': [
                    [top_left_x, top_left_y, bottom_right_x, bottom_right_y, confidence, class_name]
                ]
            }
        }
    """
    dataset_dataframe = importer.ImportVOC(path=annotations_folder_path)
    return convert_dataframe_to_dict(dataset_dataframe, annotations_folder_path, classes_list)


def convert_dataframe_to_dict(dataframe, annotations_folder_path, classes_list = ["car", "person", "motorbike", "truck"]):
    """
    Convert a dataset dataframe to a dictionary format.

    Args:
        dataframe (pd.DataFrame): DataFrame generated by the pylabel importer.
        annotations_folder_path (str): Path to the folder containing the annotation files.
        classes_list (List[str]): List of classes to be included in the dictionary. Defaults to ["car", "person", "motorbike", "truck"].

    Returns:
        dict: A dictionary where each key is an image ID and each value is a dictionary containing the image path and bounding boxes.
    """
    dataset_dict = {}

    # Filter the classes of interest
    dataframe_filtered_by_class = dataframe.df.query("cat_name in @classes_list")

    img_id_list = dataframe_filtered_by_class["img_id"].unique().tolist()

    for img_id in img_id_list:
        rows = dataframe_filtered_by_class[dataframe_filtered_by_class["img_id"] == img_id]
        img_name = rows["img_filename"].values[0]
        img_path = os.path.join(annotations_folder_path, img_name)
        bboxes = []
        for _, row in rows.iterrows():
            top_left_x = int(row["ann_bbox_xmin"])
            top_left_y = int(row["ann_bbox_ymin"])
            bottom_right_x = int(row["ann_bbox_xmax"])
            bottom_right_y = int(row["ann_bbox_ymax"])
            confidence = 1.0
            class_name = row["cat_name"]
            bboxes.append(
                [
                    top_left_x,
                    top_left_y,
                    bottom_right_x,
                    bottom_right_y,
                    confidence,
                    class_name
                ]
            )

        dataset_dict[img_name.split(".")[0]] = {
            "img_path": img_path,
            "bboxes": bboxes,
        }

    return dataset_dict

def get_coco_name_from_id(class_id):
    """
    Retrieve the COCO class name given a class ID.

    Args:
        class_id (int): The ID of the class.

    Returns:
        str: The name of the class corresponding to the given ID.
    """
    id_to_name = {
        0: 'person',
        1: 'bicycle',
        2: 'car',
        3: 'motorbike',
        4: 'airplane',
        5: 'bus',
        6: 'train',
        7: 'truck',
        8: 'boat',
        9: 'traffic light',
        10: 'fire hydrant',
        11: 'stop sign',
        12: 'parking meter',
        13: 'bench',
        14: 'bird',
        15: 'cat',
        16: 'dog',
        17: 'horse',
        18: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra',
        23: 'giraffe',
        24: 'backpack',
        25: 'umbrella',
        26: 'handbag',
        27: 'tie',
        28: 'suitcase',
        29: 'frisbee',
        30: 'skis',
        31: 'snowboard',
        32: 'sports ball',
        33: 'kite',
        34: 'baseball bat',
        35: 'baseball glove',
        36: 'skateboard',
        37: 'surfboard',
        38: 'tennis racket',
        39: 'bottle',
        40: 'wine glass',
        41: 'cup',
        42: 'fork',
        43: 'knife',
        44: 'spoon',
        45: 'bowl',
        46: 'banana',
        47: 'apple',
        48: 'sandwich',
        49: 'orange',
        50: 'broccoli',
        51: 'carrot',
        52: 'hot dog',
        53: 'pizza',
        54: 'donut',
        55: 'cake',
        56: 'chair',
        57: 'couch',
        58: 'potted plant',
        59: 'bed',
        60: 'dining table',
        61: 'toilet',
        62: 'tv',
        63: 'laptop',
        64: 'mouse',
        65: 'remote',
        66: 'keyboard',
        67: 'cell phone',
        68: 'microwave',
        69: 'oven',
        70: 'toaster',
        71: 'sink',
        72: 'refrigerator',
        73: 'book',
        74: 'clock',
        75: 'vase',
        76: 'scissors',
        77: 'teddy bear',
        78: 'hair drier',
        79: 'toothbrush'
    }

    return id_to_name.get(class_id, "Unknown class ID")