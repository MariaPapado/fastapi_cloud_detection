from fastapi import FastAPI
import uvicorn
import cv2
from pydantic import BaseModel
from pimsys.regions.RegionsDb import RegionsDb
import os
import torch
import cv2
import json
import base64
import numpy as np
from shapely import geometry
import myUNF
import rasterio
import requests


def load_tif_image(layer):
    # Define mapserver path
    mapserver_path = '/cephfs/mapserver/data'
    # Get path to after image
    layer_datetime = layer['capture_timestamp']
    path_tif = '/'.join([mapserver_path, layer_datetime.strftime('%Y%m%d'), layer['wms_layer_name']]) + '.tif'
#    print(path_tif)
    # Load image
    tif = rasterio.open(path_tif)
    img = tif.read()
    tif.close()
    # Normalize bands on min and max
    img = normalise_bands(img)
    # Normalize
    # img = exposure.equalize_adapthist(np.moveaxis(img, 0, -1), 100)
    # img = np.moveaxis(img, -1, 0)
    # Get tif bounds
    image_bounds = list(tif.bounds)
    image_poly = geometry.Polygon.from_bounds(image_bounds[0], image_bounds[1], image_bounds[2], image_bounds[3])
    return img, tif.transform, image_bounds, image_poly


def normalise_bands(image, percentile_min=2, percentil_max=98):
    tmp = []
    for i in range(image.shape[0]):
        perc_2 = np.percentile(image[i, :, :], percentile_min)
        perc_98 = np.percentile(image[i, :, :], percentil_max)
        band = (image[i, :, :] - perc_2) / (perc_98 - perc_2)
        band[band < 0] = 0.
        band[band > 1] = 1.
        tmp.append(band)
    return np.array(tmp)

def normalize_img(img, mean=[0.467, 0.504, 0.493], std=[0.306, 0.307, 0.322]):  #

    img_array = np.asarray(img, dtype=np.float32)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img

def postprocess(image):
    # Find contours
    contours_first, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = []
    for con in contours_first:
        area = cv2.contourArea(con)
        if area>150:
            contours.append(con)

    output = cv2.drawContours(np.zeros((image.shape[0], image.shape[1],3)), contours, -1, (255,255,255), thickness=cv2.FILLED)

    # Smooth the mask
    blurred_mask = cv2.GaussianBlur(output, (25, 25), 0)

    # Threshold back to binary
    _, smoothed_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

    return smoothed_mask

def pad_left(arr, n=256):
    deficit_x = n - arr.shape[1] % n
    deficit_y = n - arr.shape[2] % n
    if not (arr.shape[1] % n):
        deficit_x = 0
    if not (arr.shape[2] % n):
        deficit_y = 0
    arr = np.pad(arr, ((0, 0), (deficit_x, 0), (deficit_y, 0)), mode="reflect")
    return arr, deficit_x, deficit_y


coords = [-61.648133585982364, 10.189181664489293]

config_db = {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
    }

database = RegionsDb(config_db)
layer_1 = database.get_optical_images_containing_point_in_period(coords, [1666631662 - 1000, 1666631662 + 1000])[0]
database.close()

image_before, tif_transform, image_bounds, image_poly = load_tif_image(layer_1)
image_before = image_before.transpose((1,2,0))
image_before = image_before[:,:,:3]
cv2.imwrite('image_before.png', image_before[:,:,[2,1,0]]*255)






image_before = (image_before).astype(np.float32)[:,:,:3]
print('shape', image_before.shape)
image_before = normalize_img(image_before)

image_before = np.transpose(image_before, (2,0,1))
image_before, dx, dy = pad_left(image_before)
image_tile = torch.from_numpy(image_before).float().unsqueeze(0).cuda()

print('padded', image_before.shape)

#cv2.imwrite('img.png', image_before[:,:,[2,1,0]]*255)

model = myUNF.UNetFormer(num_classes=2)

model.load_state_dict(torch.load('vhr_cloud_net_18.pt'))
model=model.cuda()
model = model.eval()

with torch.no_grad():
    output = model(image_tile).data.cpu().numpy()[0]
#        output = np.argmax(output[[0,3,2,1]],axis=0)
output = np.argmax(output, axis=0)
output = output[dx:, dy:]
print('aaaaaaaauniiiiii', np.unique(output))
output = postprocess(output.astype(np.uint8))
output = output[:,:,0].astype(np.uint8)
print('uniiiiii', np.unique(output))
print('type', type(output[10,10]))
print(output.shape)
print(np.unique(output))
cv2.imwrite('mask.png', output)
