import io
import json
import requests


class HttpHelper:
    def __init__(self, rest_api_url="http://localhost:5000/model/predict"):
        self.rest_api_url = rest_api_url

    def post_file(self, filepath):
        payload = open(filepath, "rb").read()
        files = {"image": (filepath, payload)}
        headers = {"Accept": "application/json"}
        url = self.rest_api_url
        response = requests.post(url, files=files, headers=headers)
        responseMessage = response.text

        try:
            response_data = json.loads(response.text)
            probability = response_data["predictions"][0]["probability"]
            formatted_probability_val = "%.10f" % probability
            return formatted_probability_val
        except requests.exceptions.RequestException:
            print(responseMessage)

    def post_inmemory_image(self, image, img_name="true.png"):

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        byte_im = buf.getvalue()

        files = {"image": (img_name, byte_im)}
        headers = {"Accept": "application/json"}
        url = self.rest_api_url
        response = requests.post(url, files=files, headers=headers)
        responseMessage = response.text

        try:
            response_data = json.loads(response.text)
            probability = response_data["predictions"][0]["probability"]
            formatted_probability_val = "%.10f" % probability
            return formatted_probability_val
        except requests.exceptions.RequestException:
            print(responseMessage)
