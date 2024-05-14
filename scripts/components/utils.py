import os
import csv
import cv2, random
import numpy as np
from collections import defaultdict
from PIL import Image
# from torchvision import transforms
import statistics as st
import xml.etree.ElementTree as ET

class ImageProcessor:
    def __init__(self):
        pass

    def find_smallest_largest_bboxes(self, bboxes):
        smallest_bbox = largest_bbox = bboxes[0]
        smallest_area = largest_area = (bboxes[0][2] - bboxes[0][0]) * (bboxes[0][3] - bboxes[0][1])

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)

            if area < smallest_area:
                smallest_area = area
                smallest_bbox = bbox
            elif area > largest_area:
                largest_area = area
                largest_bbox = bbox

        return smallest_bbox, largest_bbox

    @staticmethod
    def count_bboxes_in_dir(annotation_dir):
        bbox_counts = []
    
        for filename in os.listdir(annotation_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(annotation_dir, filename), 'r') as file:
                    bbox_count = sum(1 for line in file if line.strip())
                    bbox_counts.append(bbox_count)
    
        return bbox_counts
    
    
    def calculate_dataset_size(x, N_min, N_max, k):
        return N_min + ((N_max - N_min) / 2) * (1 + 1 / (1 + np.exp(-k * x)))
    
    @staticmethod
    def filter_bboxes_by_class(bboxes_dict, class_needed):
        filtered_bboxes = {}

        if class_needed in bboxes_dict:
            filtered_bboxes[class_needed] = bboxes_dict[class_needed]
    
        return filtered_bboxes
    
    @staticmethod
    def parse_xml(xml_file_path):
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        class_names = []
        polygons = []
        bboxes = []

        # Extract image dimensions
        image_width = float(root.find('.//imagesize/ncols').text)
        image_height = float(root.find('.//imagesize/nrows').text)
        # Iterate over each object element in the XML
        for object_elem in root.findall('.//object'):
            # Get the classname
            class_name = object_elem.find('name').text if object_elem.find('name') is not None else "Unnamed"
            class_names.append(class_name)

            # Parsethe polygon points
            polygon_points = []
            if object_elem.find('polygon') is not None:
                for pt in object_elem.find('polygon').findall('pt'):
                    x = float(pt.find('x').text)  # Changed to float to handle decimal points
                    y = float(pt.find('y').text)
                    polygon_points.append((x, y))
            polygons.append(polygon_points)

            # Calculate the bounding box from polygon points
            if polygon_points:
                min_x = min(p[0] for p in polygon_points)
                max_x = max(p[0] for p in polygon_points)
                min_y = min(p[1] for p in polygon_points)
                max_y = max(p[1] for p in polygon_points)
                # Finding the center and width and hieght
                cx = ((min_x+max_x)/2)/image_width
                cy = ((min_y+max_y)/2)/image_height
                width = (max_x - min_x)/image_width
                height = (max_y - min_y)/image_height
                bboxes.append((cx, cy, width, height))
            else:
                bboxes.append((0, 0, 0, 0))  # Placeholder for objects with no polygon points

        return class_names, polygons, bboxes
    
    @staticmethod
    def parse_txt(txt_file_path):
        
        class_names = []
        polygons = []
        bboxes = []

        # Read the text file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
        count = 0

        for line in lines:
            row = line.split(" ")
            class_id = int(row[0])  # Assuming class_id is the first element and is an integer
            class_names.append(class_id)

            # Extract points (assuming they are normalized as x1, y1, x2, y2, ...)
            points = list(map(float, row[1:]))  # Convert each coordinate to float from string
            polygon_points = [(points[i], points[i+1]) for i in range(0, len(points), 2)]
            polygons.append(polygon_points)

            # Calculate the bounding box from polygon points
            if polygon_points:
                xs = [p[0] for p in polygon_points]
                ys = [p[1] for p in polygon_points]
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                bboxes.append((min_x, min_y, max_x, max_y))
            else:
                bboxes.append((0, 0, 0, 0))  # Placeholder for objects with no polygon points
            count+=1
        return class_names, polygons, bboxes

    
    @staticmethod
    def parse_annotation_line(line):
        parts = line.strip().split()
        class_num = int(parts[0])
        bbox = list(map(float, parts[1].split(',')))
        return class_num, bbox
    
    @staticmethod
    def bbox_area_normalized(bbox, image_shape):
        width, height = bbox[2], bbox[3]
        image_height, image_width = image_shape[:2]
        abs_width = width * image_width
        abs_height = height * image_height
        return abs_width * abs_height
    
    @staticmethod
    def get_bboxes_from_annotations(annotations_dir, total_classes, model_type):
        bboxes_dict = defaultdict(list)

        for filename in os.listdir(annotations_dir):
            # if model_type in ["fasterrcnn", "yolo", "pointrend"]:
            if filename.endswith(".txt"):
                image_name = filename.replace(".txt", "")
                with open(os.path.join(annotations_dir, filename), 'r') as file:
                    for line in file:
                        parts = line.strip().split()
                        class_id = int(parts[0])
                        bbox = list(map(float, parts[1:5]))
                        bboxes_dict[class_id].append((image_name, bbox))
            # elif model_type == "pointrend":
            #     if filename.endswith(".xml"):
            #         image_name = filename.replace(".xml", "")
            #         class_names, polygons, bboxes = ImageProcessor.parse_xml(xml_file_path=os.path.join(annotations_dir, filename))
            #         for class_name, poly, box in zip(class_names, polygons, bboxes):
            #             class_id = total_classes.index(class_name)
            #             bboxes_dict[class_id].append((image_name, box))

            #     elif filename.endswith(".txt"):
            #         image_name = filename.replace(".txt", "")
            #         class_names, polygons, bboxes = ImageProcessor.parse_txt(txt_file_path=os.path.join(annotations_dir, filename))
            #         for class_id, poly, box in zip(class_names, polygons, bboxes):
            #             # class_id = total_classes.index(class_name)
            #             bboxes_dict[class_id].append((image_name, box))

        return bboxes_dict

    # def preprocess_image(self, cropped_image):
    #     preprocess = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    #     ])
    #     image_tensor = preprocess(cropped_image).unsqueeze(0).to("cuda")
    #     return image_tensor
    
    @staticmethod
    def crop_image(image_path, bbox, resize=True, normalise=True, output_size=(600, 600), output_dir='dataset/outputs'):
        image = Image.open(image_path).convert('RGB')
        if normalise:
            center_x, center_y, width, height = bbox
            image_width, image_height = image.size

            half_width, half_height = width * image_width / 2, height * image_height / 2
            x1 = int((center_x * image_width) - half_width)
            y1 = int((center_y * image_height) - half_height)
            x2 = int((center_x * image_width) + half_width)
            y2 = int((center_y * image_height) + half_height)
        else:
            x1, y1, w, h = bbox
            x2, y2 = x1+w, y1+h
        
        # print("IMAGE SHAPE: ", image.size)
        # print((x1, y1, x2, y2))
        cropped_image = image.crop((x1, y1, x2, y2))
        if resize:
            cropped_image = cropped_image.resize(output_size)
        return cropped_image
    
    @staticmethod
    def save_image_from_numpy(numpy_array, output_dir, filename):
        os.makedirs(output_dir, exist_ok=True)
        image = Image.fromarray((numpy_array * 255).astype('uint8'))
        image.save(os.path.join(output_dir, filename))

    def normalize_image(self, image):
        np_image = np.array(image).astype(np.float32)
        mean = np_image.mean()
        std = np_image.std()
        normalized_image = (np_image - mean) / std
        return normalized_image
    
    @staticmethod
    def calculate_histogram(image):
        np_image = np.array(image)
        gray_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        return hist    
    


class FgBgUtils:
    
    @staticmethod
    def get_crops(boxes_dict, img_dir, get_single=False):
        crops_dict = dict()
        for key, value in boxes_dict.items():
            crops_dict[key] = []
            for img_name, box in value:
                if get_single and len(crops_dict[key])==1:
                    break
                image = ImageProcessor.crop_image(image_path=os.path.join(img_dir, f"{img_name}.png"), bbox=box, resize=False)
                crops_dict[key].append(image)
        return crops_dict
    

    @staticmethod
    def apply_template_matching(image, template, threshold):
        # print("IMAGE TYPE", type(template))
        image = np.array(image)
        template = np.array(template)
        # Convert it to grayscale
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Store width and height of template in w and h
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)

        # Convert match results to rectangles for NMS
        rectangles = FgBgUtils.convert_to_rectangles(zip(*loc[::-1]), w, h)
        scores = res[loc]

        # Apply Non-Maximum Suppression
        # The scores need to be a list of scores for each rectangle, and we're using the match confidence as the score
        # cv2.dnn.NMSBoxes requires rects in the format [x, y, width, height], confidence scores, and two NMS parameters: score threshold and nms threshold
        indices = cv2.dnn.NMSBoxes(rectangles, scores.tolist(), threshold, 0.4)

        final_boxes = [rectangles[ind] for ind in indices]
        print("FINAL TEMPLATES: ", final_boxes)
        return final_boxes


    # Function to convert template matching result to the format required by cv2.dnn.NMSBoxes
    @staticmethod
    def convert_to_rectangles(points, w, h):
        rects = []
        for pt in points:
            rects.append([int(pt[0]), int(pt[1]), w, h])
        return rects

    @staticmethod
    def get_template_matching_result(crops_dict, negative_images_dir, threshold):
        template_res = dict()
        # print(crops_dict)
        for class_id, crop in crops_dict.items():
            template_res[class_id] = []
            for img_name in os.listdir(negative_images_dir):
                img_path = os.path.join(negative_images_dir, img_name)
                img = cv2.imread(img_path)
                similar_boxes = FgBgUtils.apply_template_matching(image=img, template=crop[0], threshold=threshold)
                # print("DETECTED TEMPLATE BOXES: ", len(similar_boxes))
                template_res[class_id]+=[[img_name.split(".")[0], box] for box in similar_boxes]

        return template_res
    

    @staticmethod
    def get_similarity_metric(positive_boxes, template_boxes, images_dir, negative_dir, data_count, classes, data_sample, env_type):
        # print("###############IN CLASS intra_class_histogram_similarity")
        histogram_values = defaultdict(list)

        for class_id, bboxes in positive_boxes.items():
            print("HERE")
            if len(bboxes)==0 or len(template_boxes[class_id])==0:
                continue
            
            for data_pt1, data_pt2 in random_combination(bboxes, template_boxes[class_id], data_sample):
                
                img_name1, bbox1 = data_pt1
                img_name2, bbox2 = data_pt2

                image1_path = os.path.join(images_dir, f'{img_name1}.png')
                cropped1 = ImageProcessor.crop_image(image1_path, bbox1, resize=False)
                np_cropped1 = np.array(cropped1)
                hist1 = ImageProcessor.calculate_histogram(np_cropped1)

                image2_path = os.path.join(negative_dir, f'{img_name2}.png')
                cropped2 = ImageProcessor.crop_image(image2_path, bbox2, resize=False, normalise=False)
                np_cropped2 = np.array(cropped2)
                hist2 = ImageProcessor.calculate_histogram(np_cropped2)

                hist_similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                histogram_values[class_id].append(hist_similarity)
        
        # print("histogram_values for the FG/BG calc: ", histogram_values)
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
            N_min = int(os.getenv(f"{env_type}_FGBG_MIN"))
            N_max = int(os.getenv(f"{env_type}_FGBG_MAX"))
            # print("AVERAGE HIST: ", average_hist)
            # Map the value to the range [0, 100]
            # print("RATIO: ", ratio)
            mapped_value = np.ceil(np.round((average_hist + 1) * 0.5 * (N_max - N_min) + N_min, 0))
            data_count[f"{classes[class_id]}_positive"] += mapped_value
            data_count[f"{classes[class_id]}_negative"] += 2*mapped_value
            print(f"NEED MORE {mapped_value} IMAGES FOR CLASSES {class_id}")

        return stats, data_count
    



def random_combination(y, x, iterations):
    for _ in range(iterations):
        yield random.choice(y), random.choice(x)
    

    



