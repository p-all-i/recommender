import os
import cv2
import datetime

class DatasetProcessor:
    def __init__(self):
        """
        Initializes the dataset processor without requiring specific dataset information.
        """
        pass

    @staticmethod
    def adjust_annotation(anno_lines, roi, img_width, img_height, crop_width, crop_height):
        new_annotations = []
        x1, y1, x2, y2 = roi
        for line in anno_lines:
            parts = line.split()
            if len(parts) < 3:  # Ensure there's at least one coordinate pair
                continue
            obj_class = parts[0]
            coords = list(map(float, parts[1:]))

            # Adjust coordinates relative to the crop
            adjusted_coords = []
            for i in range(0, len(coords), 2):
                abs_x = coords[i] * img_width
                abs_y = coords[i + 1] * img_height

                # Only keep points within the ROI
                if x1 <= abs_x <= x2 and y1 <= abs_y <= y2:
                    new_x = (abs_x - x1) / crop_width
                    new_y = (abs_y - y1) / crop_height
                    adjusted_coords.extend([new_x, new_y])

            if adjusted_coords:
                new_annotations.append(f"{obj_class} " + ' '.join(map(str, adjusted_coords)) + "\n")

        return new_annotations
    
    def normalize_roi(self, roi, height, width):
        x1, y1, x2, y2 = roi

        if all(0 <= n <= 1 for n in roi):
            # Convert normalized coordinates to pixel values
            x1 = float(x1) * width
            y1 = float(y1) * height
            x2 = float(x2) * width
            y2 = float(y2) * height

        return [int(x1), int(y1), int(x2), int(y2)]
    

    def process_dataset(self, dataset_dirs, rois, output_dir):
        print("Starting dataset processing...")
        for dataset_dir in dataset_dirs:
            print(f"\nProcessing dataset: {dataset_dir}")

            if not os.path.isdir(dataset_dir):
                print(f"Error: The directory does not exist {dataset_dir}.")
                continue

            dir_name = dataset_dir.split("/")[-1]

            for roi_id, roi_info in rois.items():
                subdataset_dir_name = f"subdataset_{roi_id}"
                subdataset_img_dir = os.path.join(output_dir, subdataset_dir_name, dir_name, "images")
                subdataset_anno_dir = os.path.join(output_dir, subdataset_dir_name, dir_name, "annotations")
                os.makedirs(subdataset_img_dir, exist_ok=True)
                os.makedirs(subdataset_anno_dir, exist_ok=True)

            for img_filename in os.listdir(os.path.join(dataset_dir, "images")):
                # if file.lower().endswith('.png'):
                # img_filename = file
                anno_filename = os.path.splitext(img_filename)[0] + '.txt'
                img_path = os.path.join(dataset_dir, "images", img_filename)
                anno_path = os.path.join(dataset_dir, "annotations", anno_filename)

                if not os.path.exists(anno_path):
                    print(f"[WARNING] {datetime.datetime.now()} Annotation file '{anno_path}' not found for image '{img_filename}'.")
                    print(f"[INFO] {datetime.datetime.now()} skipping {img_filename}")
                    continue

                img = cv2.imread(img_path)
                if img is None:
                    print(f"[ERROR] {datetime.datetime.now()} Failed to read image '{img_path}'.")
                    print(f"[INFO] {datetime.datetime.now()} skipping {img_filename}")
                    continue

                img_height, img_width = img.shape[:2]

                with open(anno_path, 'r') as f:
                    anno_lines = f.readlines()

                for roi_id, roi_info in rois.items():
                    roi = roi_info["cropping"]
                    roi = self.normalize_roi(roi=roi, height=img_height, width=img_width)
                    x1, y1, x2, y2 = roi
                    crop = img[y1:y2, x1:x2]
                    if crop.size == 0:
                        print(f"[WARNING] {datetime.datetime.now()} Cropped image is empty. Check ROI coordinates.")
                        print(f"[INFO] {datetime.datetime.now()} skipping {img_filename}")
                        continue

                    crop_height, crop_width = crop.shape[:2]
                    try:
                        new_annotations = self.adjust_annotation(anno_lines, roi, img_width, img_height, crop_width, crop_height)
                    except Exception as e:
                        print(f"[ERROR] {datetime.datetime.now()} Failed to adjust annotations to roi '{img_path}' {e}.")
                        print(f"[INFO] {datetime.datetime.now()} skipping {img_filename}")
                        continue


                    subdataset_dir_name = f"subdataset_{roi_id}"
                    subdataset_img_dir = os.path.join(output_dir, subdataset_dir_name, dir_name, "images")
                    subdataset_anno_dir = os.path.join(output_dir, subdataset_dir_name, dir_name, "annotations")

                    os.makedirs(subdataset_img_dir, exist_ok=True)
                    os.makedirs(subdataset_anno_dir, exist_ok=True)

                    

                    unique_filename = os.path.splitext(os.path.basename(img_filename))[0]+f"_{roi_id}"
                    cropped_img_path = os.path.join(subdataset_img_dir, f"{unique_filename}.png")
                    cropped_anno_path = os.path.join(subdataset_anno_dir, f"{unique_filename}.txt")

                    cv2.imwrite(cropped_img_path, crop)
                    with open(cropped_anno_path, 'w') as f:
                        f.writelines(new_annotations)
                    print(f"Saved cropped image for '{img_filename}' in '{subdataset_img_dir}'.")
                    print(f"Saved cropped annotations for '{img_filename}' in '{subdataset_anno_dir}'.")

        print("\nDataset processing completed.")

        

if __name__ == "__main__":
    data_dirs = [
        "/home/frinks2/amal/assembly_training_test/data1/"
        "/home/frinks2/amal/assembly_training_test/data1/"
        # Add more dataset pairs as needed
    ]
    # rois = [[1420, 400, 2010, 980], [2050, 470, 2620, 1270]]  # Example ROIs
    rois = {
          "2_roi_id_1": {
            "classes": ["class3"],
              "cropping": [1420, 400, 2010, 980]
          },
          "2_roi_id_2": {
            "classes": ["class1", "class2"],
            "cropping": [2050, 470, 2620, 1270]
          }
        }
    output_dir = '/home/frinks2/amal/assembly_training_test/final_data'
    processor = DatasetProcessor()

    processor.process_dataset(data_dirs=data_dirs, rois=rois, output_dir=output_dir)

    # process_dataset(data_dirs, rois, output_dir)

