import torch
import torchvision.transforms as T

from PIL import Image, ImageDraw

import time

import det
import camera
import cv2


def draw(images, labels, boxes, scores, thrh = 0.6):
    for i, im in enumerate(images):
        draw = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scores[i][scr > thrh]

        for j,b in enumerate(box):
            draw.rectangle(list(b), outline='red',)
            draw.text((b[0], b[1]), text=f"{lab[j].item()} {round(scrs[j].item(),2)}", fill='green', )

        im.save(f'results_{i}.png')


def main():
    '''
    im_file = R'./test.png'
    device = 'cpu' # cpu, cuda:0

    print('DET Model Init ----------------------------------------')

    model = det.Model().to(device)

    print('DET Model Init Done')


    print('Runtime ----------------------------------------')

    im_pil = Image.open(im_file).convert('RGB')
    w, h = im_pil.size
    orig_size = torch.tensor([w, h])[None].to(device)

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    im_data = transforms(im_pil)[None].to(device)

    tStart = time.time()
    output = model(im_data, orig_size)
    tEnd = time.time()
    print(f'Inference time {tEnd - tStart}')

    labels, boxes, scores = output

    draw([im_pil], labels, boxes, scores)
    '''
    loc = camera.Localize()

    debugImg = cv2.imread('test3.png')
    loc.localize(debugImg)

    cv2.imshow('Debug', debugImg)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()