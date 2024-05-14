import requests
import time

class Poster:
    def __init__(self, url, max_retries=3, delay=2):
        """
        Initializes the RetriablePoster instance.

        Parameters:
        - url (str): The URL to which the data will be posted.
        - max_retries (int): Maximum number of retries if the request fails.
        - delay (int): Delay in seconds before retrying the request.
        """
        self.url = url
        self.max_retries = max_retries
        self.delay = delay

    def post(self, data):
        """
        Attempts to POST data to the specified URL with retries on failure.

        Parameters:
        - data (dict): The data to be posted to the URL.

        Returns:
        - response (requests.Response): The response object from the server.

        Raises:
        - Exception: Raises an exception if the maximum number of retries is exceeded.
        """
        attempts = 0
        while attempts < self.max_retries:
            try:
                response = requests.post(self.url, json=data)
                response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
                return response
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempts + 1} failed: {e}")
                attempts += 1
                if attempts < self.max_retries:
                    time.sleep(self.delay)
                else:
                    raise Exception(f"Failed to POST data after {self.max_retries} attempts.") from e

        return None
