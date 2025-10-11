print('Init -------------------------------------------')
import cv2
import numpy as np
import torch
import torchvision.transforms as T

from PIL import Image

import subprocess
import socket
import struct

import det
import camera

def main():
    DEBUG_PORT = 25565
    CAMERA_IMAGE = 'camera_capture.png'
    DEBUG_IMAGE = 'debug_capture.png'
    DEVICE = 'cpu' # cpu, cuda:0
    INF_TH = 0.5 # Inference threshold

    print('Load DET Model')
    model = det.Model().to(DEVICE)

    print('Load localization')
    imgPt = np.array([
        [0, 0],
        [0, 0],
        [0, 0]
    ], np.float32)
    loc = camera.Localize()

    print('Runtime ----------------------------------------')
    iteration = 0
    while True:
        # Capture image from camera
        subprocess.run([
            'rpicam-still',
            '--width', '1920',
            '--height', '1080',
            '--encoding', 'png',
            '--output', CAMERA_IMAGE], stdout=subprocess.DEVNULL)
        print(f'[{iteration}] Captured image')

        # Run inference
        im_pil = Image.open(CAMERA_IMAGE).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([w, h])[None].to(DEVICE)

        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        im_data = transforms(im_pil)[None].to(DEVICE)

        output = model(im_data, orig_size)
        labels, boxes, scores = output
        print(f'[{iteration}] Ran inference')


        # Coordinate transform
        debugImg = cv2.imread(CAMERA_IMAGE)
        deprojPoint = [376, 518]
        loc.localize(imgPt, deprojPoint, debugImg)


        # Draw inference boxes
        scr = scores[0]
        lab = labels[0][scr > INF_TH]
        box = boxes[0][scr > INF_TH]
        scrs = scores[0][scr > INF_TH]
        for _, b in enumerate(box):
            xy = b.detach().numpy().astype(np.int32)
            cv2.rectangle(debugImg, xy[0:2], xy[2:4], (0, 0, 255), 2)

        cv2.imwrite(DEBUG_IMAGE, debugImg)
        # cv2.imshow('Debug', debugImg)
        # cv2.waitKey(0)
        print(f'[{iteration}] Write debug image')


        # Handle debug messages
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as dbgSocket:
                dbgSocket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                dbgSocket.settimeout(1)
                dbgSocket.bind(('', DEBUG_PORT))
                dbgSocket.listen(1)
                conn, addr = dbgSocket.accept()
                with conn:
                    # Receive config
                    cfgBytes = conn.recv(24)
                    x0, y0, x1, y1, x2, y2 = struct.unpack('>IIIIII', cfgBytes)
                    imgPt = np.array([
                        [x0, y0],
                        [x1, y1],
                        [x2, y2]
                    ], np.float32)

                    # Read the debug image and send it
                    with open(DEBUG_IMAGE, 'rb') as file:
                        imgBytes = file.read()
                    conn.sendall(imgBytes)
                print(f'[{iteration}] Update debug connection')
        except socket.timeout:
            print(f'[{iteration}] Debug connection timeout')
        except (ConnectionAbortedError, ConnectionResetError):
            print(f'[{iteration}] Debug connection aborted')
        except struct.error:
            print(f'[{iteration}] Failed to parse debug config')

        iteration += 1

if __name__ == '__main__':
    main()