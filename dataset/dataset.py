from pylabel import importer
import pandas as pd
import csv
import os


class Dataset:
    """ Core fast-openISP pipeline """

    def __init__(self):
        pass

    def VOC_dataset(self, annotations_folder, classes=["car"]):
        dataset_dict = {}
        dataset = importer.ImportVOC(path=annotations_folder, path_to_images=annotations_folder)
        filtered_df = dataset.df.query('cat_name in @classes')

        img_id_list = filtered_df['img_id'].unique().tolist()
        for img_id in img_id_list:
            rows = filtered_df[filtered_df['img_id'] == img_id]
            # Obtain image path
            img_name = rows['img_filename'].values[0]
            img_path = os.path.join(rows['img_folder'].values[0], img_name)
            bboxes = []
            for _, row in rows.iterrows():
                # [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
                top_left_x = int(row['ann_bbox_xmin'])
                top_left_y = int(row['ann_bbox_ymin'])
                bottom_right_x = int(row['ann_bbox_xmax'])
                bottom_right_y = int(row['ann_bbox_ymax'])
                cls = row['cat_name']
                #width = int(row['ann_bbox_xmax']) - top_left_x
                #height = int(row['ann_bbox_ymax']) - top_left_y
                bboxes.append([top_left_x, top_left_y, bottom_right_x, bottom_right_y, cls])


            dataset_dict[img_name.split('.')[0]] = {'img_path': img_path, 'bboxes': bboxes}
        return dataset_dict

    def RAW_cars_dataset_ann(self, csv_path):
        bboxes_dict = {}
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader)  # skip header row

            for row in reader:
                frame_name = "{:03d}".format(int(row[0]))
                if frame_name not in bboxes_dict:
                    bboxes_dict[frame_name] = []

                for i in range(1, 60, 4):
                    if row[i] == "":
                        break
                    else:
                        # [top_left_x, top_left_y, bottom_right_x, bottom_right_y]
                        top_left_x = int(row[i])
                        top_left_y = int(row[i + 1])
                        bottom_right_x = top_left_x+int(row[i + 2])
                        bottom_right_y = top_left_y+int(row[i + 3])
                        #width = int(row[i + 2])
                        #height = int(row[i + 3])

                        bboxes_dict[frame_name].append([top_left_x, top_left_y, bottom_right_x, bottom_right_y])

        return bboxes_dict

    def RAW_cars_dataset(self, csv_path, imgs_folder):
        dataset_dict = {}
        bboxes_dict = self.RAW_cars_dataset_ann(csv_path)

        for file_name in os.listdir(imgs_folder):
            if file_name.endswith(".png"):
                img_path = os.path.join(imgs_folder, file_name)
                frame_name = file_name.split('.')[0]
                dataset_dict[frame_name] = {'img_path': img_path, 'bboxes': bboxes_dict[frame_name]}

        return dataset_dict