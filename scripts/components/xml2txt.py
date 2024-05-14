import os
import xml.etree.ElementTree as ET

class XMLToYOLOConverter:
    def __init__(self, class_names):
        self.class_names = class_names
        self.class_mapping = {name: idx for idx, name in enumerate(class_names)}

    def convert_directory(self, input_dir):
        # No need for output_dir, using input_dir for output as well
        for filename in os.listdir(input_dir):
            if filename.endswith('.xml'):
                input_path = os.path.join(input_dir, filename)
                # Using the same directory and name for the output, but with a different extension
                output_path = os.path.splitext(input_path)[0] + '.txt'
                self.convert_file(input_path, output_path)
                # After successful conversion, delete the XML file
                os.remove(input_path)
                print(f"Converted and deleted: {input_path}")

    def convert_file(self, input_path, output_path):
        tree = ET.parse(input_path)
        root = tree.getroot()

        image_size = root.find('imagesize')
        image_width = float(image_size.find('ncols').text)
        image_height = float(image_size.find('nrows').text)

        annotations = []

        for obj in root.iter('object'):
            obj_name = obj.find('name').text
            class_id = self.class_mapping.get(obj_name)

            polygon_points = []
            for polygon in obj.iter('polygon'):
                for pt in polygon.iter('pt'):
                    x = float(pt.find('x').text) / image_width
                    y = float(pt.find('y').text) / image_height
                    polygon_points.extend([x, y])

            annotation_line = f"{class_id} " + " ".join(map(str, polygon_points))
            annotations.append(annotation_line)

        with open(output_path, 'w') as f:
            f.write("\n".join(annotations))


def convert_xml(data_dir, classes):
    # Creating a Class object
    convertor = XMLToYOLOConverter(class_names=classes)
    for dir_name in os.listdir(data_dir):
        dir_path = os.path.join(data_dir, dir_name, "annotations")
        convertor.convert_directory(input_dir=dir_path)
