import os
import requests
import time, datetime
from concurrent.futures import ThreadPoolExecutor
import uuid
import os
import boto3




class S3Downloader:
    def __init__(self, s3_resource, max_workers=5):
        """
        Initializes the downloader with an S3 resource, a target directory for downloads, and the maximum number of worker threads.

        :param s3_resource: boto3 S3 resource object.
        :param target_directory: Path to the directory where files should be saved.
        :param max_workers: Maximum number of threads for parallel downloads.
        """
        self.s3 = s3_resource
        self.max_workers = max_workers
        

    def _parse_s3_url(self, s3_url):
        # Parsing logic remains the same...
        """
        Parses an S3 URL to extract the bucket name and object key.

        :param s3_url: S3 URL to parse.
        :return: Tuple of (bucket_name, object_key).
        """
        if s3_url.startswith("s3://"):
            s3_url = s3_url[5:]
        bucket_name, object_key = s3_url.split("/", 1)
        return bucket_name, object_key

    def _download_file(self, s3_url, local_filename, target_directory):
        print("downloading...")
        # Download logic remains the same...
        """
        Downloads a single file from S3 to the local filesystem.

        :param s3_url: S3 URL of the file to download.
        :param local_filename: Filename to save the file as locally.
        """
        bucket_name, object_key = self._parse_s3_url(s3_url)
        local_path = os.path.join(target_directory, local_filename)
        self.s3.Bucket(bucket_name).download_file(object_key, local_path)
        print(f"Downloaded {local_path}")
        

    def download_pairs(self, pairs, target_directory):
        os.makedirs(target_directory, exist_ok=True)

        # Image directory
        img_dir = os.path.join(target_directory, "images")
        anno_dir = os.path.join(target_directory, "annotations")

        """
        Downloads image-annotation pairs in parallel, ensuring unique and consistent naming.

        :param pairs: List of dictionaries with "image" and "annotation" S3 URLs.
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for pair in pairs:
                unique_id = str(uuid.uuid4())
                image_ext = os.path.splitext(pair["image"])[1]
                annotation_ext = os.path.splitext(pair["annotation"])[1]

                image_filename = f"{unique_id}{image_ext}"
                annotation_filename = f"{unique_id}{annotation_ext}"

                executor.submit(self._download_file, pair["image"], image_filename, img_dir)
                executor.submit(self._download_file, pair["annotation"], annotation_filename, anno_dir)
        return annotation_ext


class getConfigData:
    """
    A class to fetch configuration data from a provided endpoint.
    """
    def __init__(self):
        self.data_dir = os.getenv("DATA_DIR")
        self.s3_resource = boto3.resource('s3')

    @staticmethod
    def getData(url, job_id):
        """
        A static method to retrieve configuration data from a predefined endpoint.
        
        Args:
        - loggerObj (object): Logger object to log information and errors.
        
        Returns:
        - dict: Dictionary containing configuration data fetched from the endpoint.
        """
        while True:  # Start an infinite loop for retry mechanism
            try:
                # Fetching data from the predefined endpoint
                # preparing the params
                params = {'jobId': job_id}
                response = requests.get(url=f"{url}", params=params)
                response.raise_for_status()  # Will raise an exception for HTTP error codes
                CONFIG_JSON = response.json()["data"]
                
                # Logging the fetched configuration data
                logger_message = f"Config data received \n\n {' - - - '*9}"
                print(f"[INFO] {datetime.datetime.now()} {logger_message}")
                # loggerObj.logger.info(logger_message)
                
                return CONFIG_JSON  # Break the loop and return if successful
            except Exception as e:
                # Log the exception details
                # traceback.print_exception(*sys.exc_info())
                logger_message = f'Exception occurred while receiving config data: {e}'
                print(f"[ERROR] {datetime.datetime.now()} {logger_message}")
                # loggerObj.logger.error(logger_message)
                print(f"[ERROR] {datetime.datetime.now()} Server not running")
                # loggerObj.logger.error("Server not running")
            
            # Wait for 5 seconds before retrying
            time.sleep(5)
        
    # Download the pairs and give the local path
    def download_pairs(self, config_data, type_of_pairs):
        pairs = config_data[type_of_pairs]
        save_path = os.path.join(self.data_dir, type_of_pairs)
        os.makedirs(save_path, exist_ok=True)

        # Creating instance of S3 Downloader
        downloader = S3Downloader(s3_resource=self.s3_resource)
        anno_extension = downloader.download_pairs(pairs=pairs, target_directory=save_path)
        return anno_extension
    
    # Processing the config data
    def process_configdata(self, config_data):
        
        # Downloading the data
        anno_extension = self.download_pairs(config_data=config_data, type_of_pairs="positive")
        self.download_pairs(config_data=config_data, type_of_pairs="negative")

        config_data["data_dir"] = self.data_dir
        return config_data, anno_extension
    
    # Main function
    def run(self, url, job_id):
        # Getting the config data
        config_data = getConfigData.getData(url=url, job_id=job_id)
        print(f"[INFO] {datetime.datetime.now()} Config DATA Received!!!")
        try:
            print(f"[INFO] {datetime.datetime.now()} Downloading Seed data")
            config_data, anno_extension = self.process_configdata(config_data=config_data)
            print(f"[INFO] {datetime.datetime.now()} Seed data Downloaded!!!")
        except Exception as e:
            print(f"[ERROR] {datetime.datetime.now()} Exception raised while trying to download the data:  {e}")
            print("Exiting the service!!!")
            exit()

        return config_data, anno_extension

    


        
    