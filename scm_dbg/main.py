from tkinter import *
from tkinter import ttk

import socket
import struct

IMG_SCALE = 2

class UI:
    def __init__(self):
        root = Tk()
        root.title('SCM Debug')

        # Top level grid
        frm = ttk.Frame(root, padding=10)
        frm.grid()

        # The left side config frame
        cfgFrm = ttk.Frame(frm)
        cfgFrm.grid(sticky=N, column=0, row=0)

        # Localization settings
        grpFrm = ttk.LabelFrame(cfgFrm, text='Localization')
        grpFrm.grid(sticky=W, column=0, row=0)

        ttk.Label(grpFrm, text='Top Left').grid(sticky=W, column=0, row=0)
        ttk.Label(grpFrm, text='Top Right').grid(sticky=W, column=0, row=1)
        ttk.Label(grpFrm, text='Bottom Left').grid(sticky=W, column=0, row=2)

        locEntry0x = StringVar()
        locEntry0y = StringVar()
        ttk.Entry(grpFrm, width=6, textvariable=locEntry0x).grid(column=1, row=0)
        ttk.Entry(grpFrm, width=6, textvariable=locEntry0y).grid(column=2, row=0)

        locEntry1x = StringVar()
        locEntry1y = StringVar()
        ttk.Entry(grpFrm, width=6, textvariable=locEntry1x).grid(column=1, row=1)
        ttk.Entry(grpFrm, width=6, textvariable=locEntry1y).grid(column=2, row=1)

        locEntry2x = StringVar()
        locEntry2y = StringVar()
        ttk.Entry(grpFrm, width=6, textvariable=locEntry2x).grid(column=1, row=2)
        ttk.Entry(grpFrm, width=6, textvariable=locEntry2y).grid(column=2, row=2)

        ttk.Button(grpFrm, width=3, text='.', command=lambda: self.pickButtonClick(0)).grid(column=3, row=0)
        ttk.Button(grpFrm, width=3, text='.', command=lambda: self.pickButtonClick(1)).grid(column=3, row=1)
        ttk.Button(grpFrm, width=3, text='.', command=lambda: self.pickButtonClick(2)).grid(column=3, row=2)

        # Information
        grpFrm = ttk.LabelFrame(cfgFrm, text='Info')
        grpFrm.grid(sticky=W, column=0, row=1)

        ttk.Label(grpFrm, text='A').grid(sticky=W, column=0, row=0)
        ttk.Label(grpFrm, text='B').grid(sticky=W, column=0, row=1)


        # The right side image preview
        img = PhotoImage()
        img = img.subsample(IMG_SCALE)
        imgLabel = ttk.Label(frm, image=img)
        imgLabel.grid(column=1, row=0)
        imgLabel.bind('<Motion> <Button-1>', lambda e: self.imgClick(e))

        pickLabel = ttk.Label(frm, text='Pick point')


        # Class data
        self.root = root # Root UI
        self.img = img   # Preview image
        self.imgLabel = imgLabel
        self.pickLabel = pickLabel # Indicates pick mode

        self.locEntry = [[locEntry0x, locEntry0y], [locEntry1x, locEntry1y], [locEntry2x, locEntry2y]]
        self.locPickMode = False
        self.locPickIndex = 0

    def update(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(0.2)
                s.connect(('127.0.0.1', 25565))

                # Send config
                for i in range(3):
                    for j in range(2):
                        if not self.locEntry[i][j].get().isnumeric():
                            self.locEntry[i][j].set('0')
                cfgBytes = struct.pack('>IIIIII',
                    int(self.locEntry[0][0].get()), int(self.locEntry[0][1].get()),
                    int(self.locEntry[1][0].get()), int(self.locEntry[1][1].get()),
                    int(self.locEntry[2][0].get()), int(self.locEntry[2][1].get()))
                s.sendall(cfgBytes)

                # Receive image
                imageBytes = bytearray()
                while True:
                    data = s.recv(4096)
                    if not data:
                        break
                    imageBytes = imageBytes + data

                with open('last_image.png', 'wb') as f:
                    f.write(imageBytes)

            # Update image
            img = PhotoImage(file='last_image.png')
            img = img.subsample(IMG_SCALE)
            self.img = img
            self.imgLabel.configure(image=self.img)
        except (socket.timeout, ConnectionAbortedError, ConnectionResetError):
            pass

    # Events
    def pickButtonClick(self, index):
        # Click event for image will set the entry
        self.locPickMode = True
        self.locPickIndex = index
        self.pickLabel.grid(sticky=(N,W), column=1, row=0)

    def imgClick(self, e):
        # Fill out the localization entry boxes
        if self.locPickMode:
            self.locPickMode = False
            self.pickLabel.grid_remove()
            self.locEntry[self.locPickIndex][0].set(str(e.x * IMG_SCALE))
            self.locEntry[self.locPickIndex][1].set(str(e.y * IMG_SCALE))

def main():
    ui = UI()
    def uiUpdate():
        ui.update()
        ui.root.after(1000, uiUpdate)
    uiUpdate()
    ui.root.mainloop()

if __name__ == '__main__':
    main()