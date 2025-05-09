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



coords = [-61.648133585982364, 10.189181664489293]

config_db = {
        "host": "sar-ccd-db.orbitaleye.nl",
        "port": 5433,
        "user": "postgres",
        "password": "sarccd-db",
        "database": "sarccd2"
    }


addr = 'http://0.0.0.0:8001'
test_url = addr + '/predict'
#test_url = 'http://cloud-detection.stg.orbitaleye-dev.nl/predict'


database = RegionsDb(config_db)
layer_1 = database.get_optical_images_containing_point_in_period(coords, [1666631662 - 1000, 1666631662 + 1000])[0]
database.close()

image_before, tif_transform, image_bounds, image_poly = load_tif_image(layer_1)
print('sh', image_before.shape)
image_before = (image_before).astype(np.float32)[:3,:,:] 
image_before = np.transpose(image_before, (1,2,0))  #(1547, 2410, 3)
width, height = image_before.shape[0], image_before.shape[1]
print('wh', width, height)
image_before_d = base64.b64encode(np.ascontiguousarray(image_before).tobytes()).decode("utf-8")

response = requests.post(test_url, json={'imageData':image_before_d, 'width': width, 'height': height})


if response.ok:
    print('ok')
    response_result = json.loads(response.text)
    response_result_data = base64.b64decode(response_result['result'])
    result = np.frombuffer(response_result_data,dtype=np.uint8)
    print('rrr', result.shape, np.unique(result))
    res = result.reshape(image_before.shape[:2])
else:
    print('no')

cv2.imwrite('res.png', res)

'''
print(image_before.shape)

image_before = (image_before).astype(np.float32) [:3,:,]
print(image_before.shape)
image_before = normalize_img(image_before*255)

#image_before = np.transpose(image_before, (2,0,1))
image_before, dx, dy = pad_left(image_before)
image_tile = torch.from_numpy(image_before).float().unsqueeze(0).cuda()

print('padded', image_before.shape)

#cv2.imwrite('img.png', image_before[:,:,[2,1,0]]*255)

model = myUNF.UNetFormer(num_classes=2)

model.load_state_dict(torch.load('vhr_cloud_net_18.pt'))
model=model.cuda()
model = model.eval()


output = model(image_tile).data.cpu().numpy()[0]
#        output = np.argmax(output[[0,3,2,1]],axis=0)
output = np.argmax(output, axis=0)
output = output[dx:, dy:]

print(output.shape)
print(np.unique(output))
cv2.imwrite('mask.png', output*255)
'''
