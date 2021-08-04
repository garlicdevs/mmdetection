# Still need to convert to COCO format
import json
import os
import pycocotools.mask as mask
import cv2

root = '../data/smoking/'


def polygonFromMask(maskedArr):  # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = mask.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = mask.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0]  # , [x, y, w, h], area


def convert(flag='train'):
    path = root + flag
    print('Flag:', flag)

    files = os.listdir(path)
    print('Total files:', len(files))

    # Get all image files
    file_names = []
    image_lookup = {}
    for file in files:
        if '.json' not in file:
            file_names.append(file)
    file_names.sort()

    ann_names = []
    with open(path + '/' + flag + '_coco.json', 'r') as f:
        obj = json.load(f)
    images = obj['images']
    for image in images:
        if not os.path.isfile(path + '/' + image['file_name']):
            print('File not exist', image['file_name'])
            images.remove(image)
        else:
            image_lookup[image['id']] = image['file_name']

    # Convert to float
    j = 0
    anns = obj['annotations']
    for ann in anns:
        if isinstance(ann['segmentation'], dict):
            maskedArr = mask.decode(ann['segmentation'])
            new_ann = polygonFromMask(maskedArr)
            for i in range(len(new_ann)):
                new_ann[i] = float(new_ann[i])
            ann['segmentation'] = [new_ann]
        else:
            if len(ann['segmentation'][0]) > 0:
                anns_ = ann['segmentation'][0]
                for i in range(len(anns_)):
                    anns_[i] = float(anns_[i])
        if len(ann['segmentation'][0]) > 0:
            ann_names.append(image_lookup[ann['image_id']])
        j += 1

    with open(path + '/' + flag + '_coco.json', 'w') as f:
        json.dump(obj, f)

    # for file in file_names:
    #     if file not in ann_names:
    #         print(file)
    #         if os.path.isfile(path + '/' + file):
    #             os.remove(path + '/' + file)
    #         else:
    #             print('File not exists')

    print()


convert(flag='train')
convert(flag='val')
convert(flag='test')


