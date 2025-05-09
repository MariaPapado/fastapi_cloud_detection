# from flask import Flask, jsonify, request
from fastapi import FastAPI, HTTPException, Request, status
import uvicorn

from fastapi.responses import JSONResponse
import traceback

from pydantic import BaseModel

import os
import torch
import cv2
import json
import base64
import numpy as np

import myUNF

# MODELS_PORT = os.environ["MODELS_PORT"]


app = FastAPI()


class HealthCheck(BaseModel):
    """Response model to validate and return when performing a health check."""

    status: str = "OK"


@app.get(
    "/healthcheck",
    tags=["healthcheck"],
    summary="Perform a Health Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=HealthCheck,
)
def get_health() -> HealthCheck:
    return HealthCheck(status="OK")


class MachineLearningModel:
    model = None

    def __init__(self, model_path):

        self.model = myUNF.UNetFormer(num_classes=2)

        #        if init_extra_channel:
        #            self.model.backbone.conv1 = torch.nn.Conv2d(
        #                4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        #            )

        self.model.load_state_dict(
            torch.load(model_path + "net_23.pt", map_location=torch.device("cpu"))
        )
        self.model = self.model.eval()

    #    def returnImage(self, image):
    #        print("return image")
    #        return image

    def pad_left(self, arr, n=256):
        deficit_x = n - arr.shape[1] % n
        deficit_y = n - arr.shape[2] % n
        if not (arr.shape[1] % n):
            deficit_x = 0
        if not (arr.shape[2] % n):
            deficit_y = 0
        arr = np.pad(arr, ((0, 0), (deficit_x, 0), (deficit_y, 0)), mode="reflect")
        return arr, deficit_x, deficit_y

    #    def normalize_img(self, img, mean=[118.974, 128.536, 125.832], std=[78.089, 78.305, 82.140]):  #
    def normalize_img(
        self, img, mean=[0.454, 0.493, 0.482], std=[0.300, 0.300, 0.316]
    ):  #

        img_array = np.asarray(img, dtype=np.float32)
        normalized_img = np.empty_like(img_array, np.float32)

        for i in range(3):  # Loop over color channels
            normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]

        return normalized_img

    def postprocess(self, image):
        # Find contours
        contours_first, hierarchy = cv2.findContours(
            image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contours = []
        for con in contours_first:
            area = cv2.contourArea(con)
            if area > 200:
                contours.append(con)

        output = cv2.drawContours(
            np.zeros((image.shape[0], image.shape[1], 3)),
            contours,
            -1,
            (255, 255, 255),
            thickness=cv2.FILLED,
        )

        # Smooth the mask
        blurred_mask = cv2.GaussianBlur(output, (25, 25), 0)

        # Threshold back to binary
        _, smoothed_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

        return smoothed_mask

    def prediction_step(self, image, threshold=0.5):
        image_tile = torch.from_numpy(image).float().unsqueeze(0)
        with torch.no_grad():
            output = self.model(image_tile).data.cpu().numpy()[0]
        #        output = np.argmax(output[[0,3,2,1]],axis=0)
        output = np.argmax(output, axis=0)
        # print(output.shape)
        return output.astype(np.uint8)

    def predict_clouds(self, img):
        img = self.normalize_img(img)
        #        width = int(img.shape[1] * scale_percent / 100)
        #        height = int(img.shape[0] * scale_percent / 100)
        #        dim = (width, height)

        x_arr = img.copy()

        # x_arr = np.transpose(img, [1, 2, 0])
        #        x_arr = cv2.resize(x_arr, dim, interpolation=cv2.INTER_AREA)
        x_arr = np.transpose(x_arr, [2, 0, 1])
        im_test, def_x, def_y = self.pad_left(x_arr)
        result = self.prediction_step(im_test)

        result = result[def_x:, def_y:]
        result = self.postprocess(result.astype(np.uint8))
        result = result[:, :, 0].astype(np.uint8)

        #        result = cv2.resize(
        #            result, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST
        #        )

        return result


###############################################################################################################


def validate_image(encoded_image: str, width: int, height: int) -> np.ndarray:
    #    decoded_image = np.fromstring(base64.b64decode(encoded_image), dtype=np.float32)
    decoded_image = np.frombuffer(base64.b64decode(encoded_image), dtype=np.float32)

    decoded_image = decoded_image.reshape(width, height, 3)
    decoded_image = np.ascontiguousarray(decoded_image)
    return decoded_image


class request_body(BaseModel):
    imageData: str
    width: int
    height: int


@app.post("/predict")
def predict(data: request_body):
    h, w = data.height, data.width
    image = validate_image(data.imageData, w, h)
    cloud_class = MachineLearningModel("/api/")
    resultData = cloud_class.predict_clouds(image)
    resultData = np.ascontiguousarray(resultData)
    result_base64 = base64.b64encode(resultData.tobytes()).decode("utf-8")
    return {"result": result_base64}


# Custom exception handler to show detailed errors
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    error_message = traceback.format_exc()
    print(error_message)  # This ensures the full error is logged in Docker logs
    return JSONResponse(
        status_code=500, content={"error": str(exc), "traceback": error_message}
    )


# if __name__ == "__main__":
#    AppScope.modelRGB = MachineLearningModel("/api/", init_extra_channel=False)
#    app.run(host="0.0.0.0", debug=True, port=MODELS_PORT)

# @app.get('/')
# def main():
