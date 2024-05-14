import os, cv2, time, datetime
from dataprocessor.PRdataProcessor import DatasetProcessor as prProcessor
from dataprocessor.ODdataProcessor import DatasetProcessor as odProcessor


class Processor:
    def __init__(self, model_type):
        self.model_type = model_type
        # if self.model_type in ["fasterrcnn", "yolo"]:
        self.processor = odProcessor()
        # elif self.model_type=="pointrend":
            # self.processor = prProcessor()
        

    def process(self, data_dir, roi_info, output_dir):

        # Iterating thorugh the positive and negative
        for dir_name in os.listdir(data_dir):
            dir_path = os.path.join(data_dir, dir_name)

            # processing the specific folder
            self.processor.process_dataset(dataset_dirs=[dir_path], rois=roi_info, output_dir=output_dir)


if __name__ == "__main__":
    dat_dir = "/home/frinks2/atharva/assembly-dataset-recommender/dataset/pointrend"
    roi_info = {
          "2-roi-id-1": {
            "classes": ["class1","class2"],
              "cropping": [1130, 190, 1890, 760]
          },
          "2-roi-id-2": {
            "classes": ["class3"],
            "cropping": [2000, 1040, 2670, 1720]
          }
        }
    output_dir = "/home/frinks2/atharva/assembly-dataset-recommender/dataset/sub_roi_res1"

    processor = Processor(model_type="pointrend")
    processor.process(data_dir=dat_dir, roi_info=roi_info, output_dir=output_dir)
            