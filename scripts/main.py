import os, sys, json
import cv2, time, datetime, traceback
import threading
import requests
from dataprocessor.dataProcessor import Processor
from components.datasetRecommender import datasetRecommender
from components.getConfig import getConfigData
from components.xml2txt import convert_xml
from components.resultposter import Poster
from dotenv import load_dotenv
load_dotenv()  # loading env variables

# Getting the config_id
CONFIG_ID = os.environ.get("CONFIG_ID")

# Downloading the data
url = os.getenv("URL")
print(f"[INFO] {datetime.datetime.now()} Getting the config DATA")
config_data, anno_extension = getConfigData().run(url=url, job_id=CONFIG_ID)
# anno_extension = "txt"

# Process the data roiswise
data = os.getenv("DATA_DIR")

modelkey = config_data["modelKey"]
# modelkey = 2

# STATES = config_data["states"]
STATES = {
    "RECOMMENDATION_STARTED": 1,
    "RECOMMENDATION_RUNNING": 2,
    "RECOMMENDATION_COMPLETED": 3,
    "RECOMMENDATION_ERROR": 4
}

class APICommunicator:
    def __init__(self, base_url):
        self.base_url = base_url
    
    def send(self, endpoint, message):
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, json=message)
        return response

    def send_error(self, config_id, error_message):
        endpoint = "error"
        message = {
            "config_id": config_id,
            "status": STATES["RECOMMENDATION_ERROR"],
            "error": error_message
        }
        self.send(endpoint, message)
total_classes = ["gear1", "bearing", "gear2"]
# roi_info = {
#           "637d1ca9-a6fc-4b33-95d2-7df7e2b88224": {
#             "classes": ["gear1","gear2"],
#               "cropping": [0, 0, 1570, 2040]
#           },
#           "637d1ca9-a6fc-4b33-95n2-7df7e2b88224": {
#             "classes": ["bearing"],
#             "cropping": [1570, 0, 3070, 2040]
#           }
#         }

api_communicator = APICommunicator(base_url=os.getenv("API_BASE_URL"))

if modelkey == 2:
    model_type = "pointrend"
elif modelkey == 1:
    model_type = "fasterrcnn"
elif modelkey == 0:
    model_type = "yolo"

print(f"[INFO] {datetime.datetime.now()} Received model type {model_type}!!!")

try:
    api_communicator.send(endpoint="status", message={"config_id": CONFIG_ID, "status": STATES["RECOMMENDATION_STARTED"]})

    print(f"[INFO] {datetime.datetime.now()} Total classes {config_data['classes']}!!!")

    # Converting the xml to txt format
    if "xml" in anno_extension: 
        print(f"[INFO] {datetime.datetime.now()} Starting to convert XML annotations to TXT format!!!")
        convert_xml(data_dir=data, classes=config_data["classes"])
        print(f"[INFO] {datetime.datetime.now()} Conversion Completed")

    roi_info = config_data["rois"]
    

    class ControlledThread(threading.Thread):
        def __init__(self, communicator, config_id, STATUS):
            super(ControlledThread, self).__init__()
            self.running = True
            self.communicator = communicator
            self.config_id = config_id
            self.TRAINING_STATUS = STATUS

        def run(self):
            while self.running:
                message = {"config_id": self.config_id, "status": self.TRAINING_STATUS}
                self.communicator.send(endpoint="status", message=message)
                print("Recommendation IS RUNNING")
                if not self.running:
                    break
                
                time.sleep(5)  # sending every 5 seconds

        def stop(self):
            self.running = False

    # Sending running status to the API
    Status_thread = ControlledThread(communicator=api_communicator, config_id=CONFIG_ID, STATUS=STATES["RECOMMENDATION_RUNNING"])
    Status_thread.start()

    output_dir = os.getenv("OUTPUT_DIR")
    print(f"[INFO] {datetime.datetime.now()} Subrois dataset saved {output_dir}")
    data_sample = int(os.getenv("DATA_SAMPLE"))

    data_processor = Processor(model_type=model_type)
    print(f"[INFO] {datetime.datetime.now()} Starting sub dataset creation")
    data_processor.process(data_dir=data, roi_info=roi_info, output_dir=output_dir)
    print(f"[INFO] {datetime.datetime.now()} Subroi dataset created successfully!!!")

    roi_wise_results = {}
    # Iterating through the sub_rois generated
    for sub_dir_name in os.listdir(output_dir):
        print(f"[INFO] {datetime.datetime.now()} Calculating and estimating dataset size for {sub_dir_name}")
        roi_id = sub_dir_name.split("_")[-1]
        # Getting the roi_info
        sub_roi_info = roi_info[roi_id]
        roi_cords = sub_roi_info["cropping"]
        roi_classes = sub_roi_info["classes"]

        # Getting the image size from the roi
        width, height = roi_cords[2] - roi_cords[0], roi_cords[3] - roi_cords[1]

        # Running the data recommender
        recommender = datasetRecommender(total_classes=total_classes, roi_classes=roi_classes, data_dir=os.path.join(output_dir, sub_dir_name), model_type=model_type, image_size=(width, height), data_sample=data_sample)
        try:
            roi_recommendation = recommender.run()
            # Send individual recommendation to API
            api_communicator.send(endpoint="recommendation", message={"config_id": CONFIG_ID, "roi_id": roi_id, "recommendation": roi_recommendation})
        except Exception as e:
            Status_thread.stop()
            Status_thread.join()
            error_message = str(e)
            api_communicator.send_error(config_id=CONFIG_ID, error_message=error_message)
            traceback.print_exception(*sys.exc_info())
            exit()

        roi_wise_results[roi_id] = roi_recommendation

    Status_thread.stop()
    Status_thread.join()

    print(f"[INFO] {datetime.datetime.now()} Final Estimation result!!!")
    print(f"[INFO] {datetime.datetime.now()}", roi_wise_results)
    # Pushing the results 
    print(roi_wise_results)

    # Posting the results to the specific endpoint
    POST_URL = os.getenv("POST_URL")
    print(f"[INFO] {datetime.datetime.now()} Trying to post Result to the specified endpoint")
    response = Poster(url=POST_URL, max_retries=5).post(data={"config_id": CONFIG_ID, "roi": roi_wise_results})
    # if response is not None:
    #     print(f"[INFO] {datetime.datetime.now()} Successfully posted the result!!!")
    # else:
    #     print(f"[ERROR] {datetime.datetime.now()} Failed to post result!!!")

    api_communicator.send(endpoint="status", message={"config_id": CONFIG_ID, "status": STATES["RECOMMENDATION_COMPLETED"], "Recommendation": roi_recommendation})
    print(f"[INFO] {datetime.datetime.now()} RECOMMENDATION SERVICE RUNNING COMPLETED!!!")

except Exception as e:
    error_message = str(e)
    api_communicator.send_error(config_id=CONFIG_ID, error_message=error_message)
    print(f"[ERROR] {datetime.datetime.now()} error occurred: {error_message}")
    traceback.print_exception(*sys.exc_info())
