import os
import cv2, time
import numpy as np
import statistics as st
from collections import defaultdict
from itertools import combinations
from components.utils import *



class datasetRecommender:
    """
    A class designed to recommend dataset sizes for AI model training by analyzing image data.
    It utilizes image processing techniques to evaluate image datasets based on annotations,
    histograms, and similarity metrics to guide the collection of a balanced and diverse dataset.
    
    Attributes:
        classes_txt_path (str): Path to a text file containing class labels, one per line.
        positive_annotations_dir (str): Directory path containing annotations for positive samples.
        negative_annotations_dir (str): Directory path containing annotations for negative samples.
        pos_images_dir (str): Directory path containing positive sample images.
        neg_image_dir (str): Directory path containing negative sample images.
        data_sample (int): Number of samples to consider for statistical calculations.
        bbox_area_normalized (function): Method from ImageProcessor to normalize bbox areas.
        count_bboxes_in_dir (function): Method from ImageProcessor to count bounding boxes in a directory.
        get_bboxes_from_annotations (function): Method from ImageProcessor to extract bounding boxes from annotation files.
        crop_image (function): Method from ImageProcessor to crop images based on bounding boxes.
        calculate_histogram (function): Method from ImageProcessor to calculate image histograms.
        calculate_dataset_size (function): Method from ImageProcessor to estimate dataset size needed.
        fg_bg_utils (FgBgUtils): Instance of FgBgUtils class for foreground-background analysis.
        classes (list): List of class labels extracted from the classes_txt_path file.
        data_count (dict): Dictionary tracking the count of data needed per class and overall.
        positive_annos (dict): Dictionary containing bounding boxes for positive samples.
        negative_annos (dict): Dictionary containing bounding boxes for negative samples.
    """
    def __init__(self, roi_classes, total_classes, data_dir, model_type, image_size, data_sample=30):
        """
        Initializes the datasetRecommender with paths to data and a sample size for analysis.
        
        Parameters:
            classes_txt_path (str): Path to the text file containing class labels.
            positive_annotations_dir (str): Path to the directory with positive annotations.
            negative_annotations_dir (str): Path to the directory with negative annotations.
            pos_images_dir (str): Path to the directory with positive images.
            neg_image_dir (str): Path to the directory with negative images.
            data_sample (int): Number of samples to use in statistical calculations. Default is 30.
        """
        # self.classes_txt_path = classes_txt_path
        self.positive_annotations_dir = os.path.join(data_dir, "positive", "annotations")
        self.negative_annotations_dir = os.path.join(data_dir, "negative", "annotations")
        self.images_dir = os.path.join(data_dir, "positive", "images")
        self.neg_image_dir = os.path.join(data_dir, "negative", "images")
        self.data_sample = data_sample
        self.image_size = image_size
        self.bbox_area_normalized = ImageProcessor.bbox_area_normalized
        self.count_bboxes_in_dir = ImageProcessor.count_bboxes_in_dir
        self.get_bboxes_from_annotations = ImageProcessor.get_bboxes_from_annotations
        self.crop_image = ImageProcessor.crop_image
        self.calculate_histogram = ImageProcessor.calculate_histogram
        self.calculate_dataset_size = ImageProcessor.calculate_dataset_size
        self.fg_bg_utils = FgBgUtils()
        self.model_type = model_type
        if self.model_type in ["fasterrcnn", "yolo"]:
            self.env_type = "OD"
        elif self.model_type == "pointrend":
            self.env_type = "PR"

        os.getenv(f"{self.env_type}_POSITIVE")
        self.classes = total_classes
        self.roi_classes = roi_classes
        self.data_count = {
            "positive": int(os.getenv(f"{self.env_type}_POSITIVE")),
            "negative": int(os.getenv(f"{self.env_type}_NEGATIVE")),
            **{f"{cls}_positive": 0 for cls in self.roi_classes},
            **{f"{cls}_negative": 0 for cls in self.roi_classes}
        }

        self.positive_annos = self.get_bboxes_from_annotations(self.positive_annotations_dir, total_classes=self.classes, model_type=self.model_type)
        self.negative_annos = self.get_bboxes_from_annotations(self.negative_annotations_dir, total_classes=self.classes, model_type=self.model_type)

    def compare_images(self, data_pt1, data_pt2):
        """
        Compares two images based on their histogram similarity.
        
        Parameters:
            data_pt1 (tuple): A tuple containing the name and bounding box of the first image.
            data_pt2 (tuple): A tuple containing the name and bounding box of the second image.
        
        Returns:
            float: The histogram similarity score between the two images.
        """
        img_name1, bbox1 = data_pt1
        img_name2, bbox2 = data_pt2

        image1_path = os.path.join(self.images_dir, f'{img_name1}.png')
        cropped1 = self.crop_image(image1_path, bbox1)
        np_cropped1 = np.array(cropped1)
        hist1 = self.calculate_histogram(np_cropped1)

        image2_path = os.path.join(self.images_dir, f'{img_name2}.png')
        cropped2 = self.crop_image(image2_path, bbox2)
        np_cropped2 = np.array(cropped2)
        hist2 = self.calculate_histogram(np_cropped2)

        hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return hist_similarity

    # Function 1: Count Classes
    def count_classes(self):
        """
        Counts the number of unique classes and calculates the additional positive images needed.
        
        Returns:
            int: The number of unique classes found, or None if an error occurs.
        """
        try:
            print("###########IN CLASS count_classes###########")
            # with open(self.classes_txt_path, 'r') as file:
            #     class_count = sum(1 for line in file if line.strip())  # Count non-empty lines

            class_count = len(self.roi_classes)
            if class_count == 1:
                print(f"Need 200 positive images.")
            if class_count > 1:
                needed_class_count = class_count-1
                print(f"Need more {100 + int(os.getenv(f'{self.env_type}_CLASSCOUNT_POSITIVE'))*needed_class_count} positive images.") 
                self.data_count["positive"] += int(os.getenv(f"{self.env_type}_CLASSCOUNT_POSITIVE"))*needed_class_count  
                self.data_count["negative"] += int(os.getenv(f"{self.env_type}_CLASSCOUNT_NEGATIVE"))*needed_class_count

            return class_count

        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    # Function 2: Instance per Image
    def instance_per_img(self):
        """
        Analyzes the mode of instance counts per image in positive annotations and adjusts
        the positive image count recommendation based on the mode.
        
        Returns:
            int: The mode of instance counts per image among positive annotations.
        """
        print("###############IN CLASS instance_per_img #################")

        def count_bboxes_per_image(bbox_dict):
            image_count = defaultdict(int)
            for images in bbox_dict.values():
                for image, _ in images:
                    image_count[image] += 1
            return list(image_count.values())

        bbox_counts = count_bboxes_per_image(self.positive_annos) 
        mode_instances = st.mode(bbox_counts)
        if mode_instances == 1:
            print("Need 200 positive images")
        if mode_instances>1:
            needed_instance_count = mode_instances-1
            print(f"Need more {100 + int(os.getenv(f'{self.env_type}_INSTPERIMG_POSITIVE'))*needed_instance_count} positive images.")
            self.data_count["positive"] += int(os.getenv(f"{self.env_type}_INSTPERIMG_POSITIVE"))*needed_instance_count
        return mode_instances


    # Function 3: Instance Min to Max Ratio
    def instance_min_max_ratio(self):
        """
        Calculates the ratio of the smallest to the largest bounding box area in positive annotations,
        and adjusts the positive image count recommendation based on this ratio.
        
        Parameters:
            image_size (tuple): The width and height of the images as a tuple (width, height).
        
        Returns:
            float: The ratio of the smallest to the largest bounding box area.
        """

        print("################IN CLASS instance_min_max_ratio ################")
        positive_annotations_dict = self.positive_annos
        all_bboxes = [bbox_data[1] for bboxes in positive_annotations_dict.values() for bbox_data in bboxes]

        if not all_bboxes:
            return None  # Return None if there are no bounding boxes

        areas = [self.bbox_area_normalized(bbox, self.image_size) for bbox in all_bboxes]

        smallest_area = min(areas)
        largest_area = max(areas)
        
        area_ratio = smallest_area / largest_area if largest_area != 0 else 0
        N_min = int(os.getenv(f"{self.env_type}_MINMAX_MIN"))
        N_max = int(os.getenv(f"{self.env_type}_MINMAX_MAX"))
        # print("AREA RATIO: ", area_ratio)
        mapped_value = np.ceil(np.round((1 - area_ratio) * (N_max - N_min) + N_min, 0))
        self.data_count["positive"] += mapped_value
        print(f"NEED MORE {mapped_value} POSITIVE IMAGES")
        return area_ratio

    # Function 4: Min Instance Img Ratio
    def min_instance_img_ratio(self):
        """
        Calculates the ratio of the smallest bounding box to the image area for each class in positive annotations,
        and adjusts the image count recommendation based on these ratios.
        
        Parameters:
            image_size (tuple): The width and height of the images as a tuple (width, height).
        
        Returns:
            dict: A dictionary with class IDs as keys and the smallest bounding box to image area ratio as values.
        """
        print("#################IN CLASS min_instance_img_ratio ################")
        bboxes_dict = self.positive_annos
        print("CLASSES ",list(bboxes_dict.keys()))
        image_area = self.image_size[0] * self.image_size[1]
        smallest_bbox_ratios = {}

        for class_id, bbox_data in bboxes_dict.items():
            smallest_area = float('inf')
            for _, bbox in bbox_data:
                area = self.bbox_area_normalized(bbox, self.image_size)
                if area < smallest_area:
                    smallest_area = area

            ratio = smallest_area / image_area if image_area else 0
            smallest_bbox_ratios[class_id] = ratio
            N_min = int(os.getenv(f"{self.env_type}_INSTIMG_MIN"))
            N_max = int(os.getenv(f"{self.env_type}_INSTIMG_MAX"))

            # Map the value to the range [0, 100]
            # print("RATIO: ", ratio)
            mapped_value = np.ceil(np.round((1 - ratio) * (N_max - N_min) + N_min, 0))
            self.data_count[f"{self.classes[class_id]}_positive"] += mapped_value
            # Print the suggested number of images for the class
            print(f"NEED MORE {mapped_value} IMAGES FOR CLASS {self.classes[class_id]}")

        return smallest_bbox_ratios

    # Function 5: Inter-Class Similarity (Histogram Only)
    def inter_class_histogram_similarity(self):
        """
        Calculates the histogram similarity between classes for a sample of data points,
        updating the image count recommendation based on the calculated similarities.
        
        Returns:
            dict: A dictionary with class pairs as keys and statistics of their histogram similarities as values.
        """
        print("###############IN CLASS inter_class_histogram_similarity ################")
        class_bboxes = self.positive_annos
        histogram_values = defaultdict(list)

        for class1, class2 in combinations(class_bboxes.keys(), 2):

            # getting data_sample amount of data points
            for data_pt1, data_pt2 in random_combination(class_bboxes[class1], class_bboxes[class2], self.data_sample):
                
                histogram_values[(class1, class2)].append(self.compare_images(data_pt1=data_pt1, data_pt2=data_pt2))

      

        stats = {}
        for class_pair in histogram_values:
            average_hist = sum(histogram_values[class_pair]) / len(histogram_values[class_pair])
            median_hist = st.median(histogram_values[class_pair])
            mode_hist = max(histogram_values[class_pair], key=histogram_values[class_pair].count, default="No unique mode")

            stats[class_pair] = {
                "histogram": {"average": average_hist, "median": median_hist, "mode": mode_hist}
            }

            N_min = int(os.getenv(f"{self.env_type}_INTERCLASS_MIN"))
            N_max = int(os.getenv(f"{self.env_type}_INTERCLASS_MAX"))
        
            dataset_size = self.calculate_dataset_size(average_hist,N_min,N_max,2)
            dataset_size = np.ceil(np.round(dataset_size,0))
            for class_id in class_pair:
                self.data_count[f"{self.classes[class_id]}_positive"] += dataset_size
            print(f"NEED MORE {dataset_size} IMAGES FOR CLASSES {class_pair}")

        return stats

    # Function 6: Intra-Class Similarity (Histogram Only)
    def intra_class_histogram_similarity(self):
        """
        Calculates the histogram similarity within each class for a sample of data points,
        updating the image count recommendation based on the calculated similarities.
        
        Returns:
            dict: A dictionary with class IDs as keys and statistics of their histogram similarities as values.
        """
        print("###############IN CLASS intra_class_histogram_similarity ################")
        class_bboxes = self.positive_annos
        histogram_values = defaultdict(list)

        for class_id, bboxes in class_bboxes.items():

            for data_pt1, data_pt2 in combinations(bboxes, 2):
                histogram_values[class_id].append(self.compare_images(data_pt1=data_pt1, data_pt2=data_pt2))

                if len(histogram_values[class_id])==self.data_sample:
                    break

        stats = {}
        for class_id in histogram_values:
            average_hist = sum(histogram_values[class_id]) / len(histogram_values[class_id])
            median_hist = st.median(histogram_values[class_id])
            try:
                mode_hist = st.mode(histogram_values[class_id])
            except st.StatisticsError:
                mode_hist = "No unique mode"

            stats[class_id] = {
                "histogram": {"average": average_hist, "median": median_hist, "mode": mode_hist}
            }

            # dataset_size = self.calculate_dataset_size(average_hist,0,200,2)
            N_min = int(os.getenv(f"{self.env_type}_INTRACLASS_MIN"))
            N_max = int(os.getenv(f"{self.env_type}_INTRACLASS_MAX"))
            # print("AVERAGE HIST: ", average_hist)
            # Map the value to the range [0, 100]
            # print("RATIO: ", ratio)
            mapped_value = N_max - np.ceil(np.round((average_hist + 1) * 0.5 * (N_max - N_min) + N_min, 0))
            self.data_count[f"{self.classes[class_id]}_positive"] += mapped_value
            print(f"NEED MORE {mapped_value} IMAGES FOR CLASSES {class_id}")

        return stats

    # Function 7: Present vs. Absent Similarity (Histogram Only)
    def present_absent_histogram_similarity(self):
        """
        Calculates the histogram similarity between present (positive) and absent (negative) instances
        for each class, updating the image count recommendation based on the calculated similarities.
        
        Returns:
            dict: A dictionary with class IDs as keys and statistics of their histogram similarities as values.
        """
        print("##############IN CLASS present_absent_histogram_similarity ################")
        positive_annotations = self.positive_annos
        negative_annotations = self.negative_annos

        similarity_stats = {}

        for class_name in self.roi_classes:
            histogram_values = []

            pos_bboxes = positive_annotations.get(self.classes.index(class_name), [])
            neg_bboxes = negative_annotations.get(self.classes.index(class_name), [])  # Fetch negative annotations for the same class
            
            if len(pos_bboxes)==0 or len(neg_bboxes)==0:
                print(class_name, "SKIPPING THIS")
                continue

            for data_pt1, data_pt2 in random_combination(pos_bboxes, neg_bboxes, self.data_sample):
                pos_img_name, pos_bbox = data_pt1
                neg_img_name, neg_bbox = data_pt2

                pos_image_path = os.path.join(self.images_dir, f'{pos_img_name}.png')  # Use the positive image directory
                pos_image = self.crop_image(pos_image_path, pos_bbox)
                hist1 = self.calculate_histogram(pos_image)

                neg_image_path = os.path.join(self.neg_image_dir, f'{neg_img_name}.png')  # Use the negative image directory
                neg_image = self.crop_image(neg_image_path, neg_bbox)
                hist2 = self.calculate_histogram(neg_image)

                hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                histogram_values.append(hist_similarity)
                # histogram_values.append(self.compare_images(data_pt1=data_pt1, data_pt2=data_pt2))

            if histogram_values:
                average_hist = sum(histogram_values) / len(histogram_values)
                median_hist = st.median(histogram_values)
                mode_hist = max(set(histogram_values), key=histogram_values.count, default="No unique mode")

                similarity_stats[self.classes.index(class_name)] = {
                    "histogram": {"average": average_hist, "median": median_hist, "mode": mode_hist}
                }

                N_min = int(os.getenv(f"{self.env_type}_PRESENTABSENT_MIN"))
                N_max = int(os.getenv(f"{self.env_type}_PRESENTABSENT_MAX"))
                # Map the value to the range [0, 100]
                # print("RATIO: ", ratio)
                mapped_value = np.ceil(np.round((average_hist + 1) * 0.5 * (N_max - N_min) + N_min, 0))
                self.data_count[f"{class_name}_positive"] += mapped_value
                self.data_count[f"{class_name}_negative"] += 2*mapped_value
                print("CURRENT CLASS: ", class_name, " MAPPED_VALUE: ", mapped_value)


        return similarity_stats
    
    ## Function8: Foreground vs  Background similarity
    def forground_background_similarity(self):
        """
        Evaluates the similarity between foreground objects in positive samples and the background in negative samples,
        adjusting the data count recommendation based on the analysis.
        
        Returns:
            dict: A dictionary with similarity statistics for foreground versus background analysis.
        """
        class_bboxes = self.positive_annos
        # Getting single crops for each class
        crops_dict = self.fg_bg_utils.get_crops(boxes_dict=class_bboxes, img_dir=self.images_dir, get_single=True)
        # Do template matching between the extracted images and negative images
        template_res = self.fg_bg_utils.get_template_matching_result(crops_dict=crops_dict, negative_images_dir=self.neg_image_dir, threshold=0.5)
        # Compare the detcted boxes with actual objects and see its similarity
        similarity_stats, self.data_count = self.fg_bg_utils.get_similarity_metric(positive_boxes=class_bboxes, template_boxes=template_res, images_dir=self.images_dir, negative_dir=self.neg_image_dir, data_count=self.data_count, classes=self.classes, data_sample=self.data_sample, env_type=self.env_type)
        return similarity_stats
    

    def get_class_count(self):
        """
        Rounds the data count recommendations to the nearest multiple of a specified number.
        
        Parameters:
            n (int): The multiple to which the data count values should be rounded.
        
        Returns:
            dict: A dictionary with rounded data count values.
        """
        def round_values_to_nearest_n(dictionary, n):
            rounded_dict = {key: round(value / n) * n for key, value in dictionary.items()}
            return rounded_dict
        return round_values_to_nearest_n(self.data_count, n=10)
    
    def run(self):
        self.count_classes()
        self.instance_per_img()
        self.instance_min_max_ratio()
        self.min_instance_img_ratio()
        self.inter_class_histogram_similarity()
        self.intra_class_histogram_similarity()
        self.present_absent_histogram_similarity()
        self.forground_background_similarity()
        return self.get_class_count()




if __name__ == "__main__":
    starting = time.time()

    # FOr HONDA
    qca = datasetRecommender('dataset/honda/classes.txt', 'dataset/honda/positive/annotations', 'dataset/honda/negative/annotations', 'dataset/honda/positive/images', 'dataset/honda/negative/images')



    # FOR KML
    # qca = QualityControlAnalyzer('dataset/classes.txt', 'dataset/positive/annotations', 'dataset/negative/annotations', 'dataset/positive/images', 'dataset/negative/images')

    qca.count_classes()

    qca.instance_per_img()

    qca.instance_min_max_ratio((3072,2048))

    qca.min_instance_img_ratio((3072,2048))

    qca.inter_class_histogram_similarity()

    qca.intra_class_histogram_similarity()

    qca.present_absent_histogram_similarity()

    qca.forground_background_similarity()

    print("FINAL DATA REQUIREMENT: ")
    print(qca.get_class_count())

    print("Time raken: ", time.time() - starting)