import time
import cv2
import numpy as np
from display import Display
from extractor import Extractor

W = 1920 // 2
H = 1080 // 2

F = 270

display = Display(W, H)
K = np.array(([F, 0, W//2], [0, F, H//2], [0, 0, 1]))
fe = Extractor(K)


def process_frame(img):
    img = cv2.resize(img, (W, H))
    matches, pose = fe.extract(img)
    if pose is None:
        return

    print("%d matches" % (len(matches)))
    print(pose)

    for pt1, pt2 in matches:
        u1, v1 = fe.denormalize(pt1)
        u2, v2 = fe.denormalize(pt2)

        cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
        cv2.line(img, (u1, v1), (u2, v2), color=(255, 0, 0))

    display.paint(img)


if __name__ == "__main__":
    cap = cv2.VideoCapture("test.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process_frame(frame)
        else:
            break


"""
FeatureExtractor(object)
#GX = 16 // 2
#GY = 12 // 2

sy = img.shape[0]//self.GY
sx = img.shape[1]//self.GX
kp_list = []
for ry in range(0, img.shape[0], sy):
    for rx in range(0, img.shape[1], sx):
        kp = self.orb.detect(img[ry:ry+sy, rx:rx+sx], None)
        for p in kp:
            p.pt = (p.pt[0] + rx, p.pt[1] + ry)
    kp_list.append(p)
return kp_list
"""


"""
process_frame(img)
for p in kp:
    u,v = map(lambda x: int(round(x)), p.pt)
    cv2.circle(img, (u,v), color=(0, 255, 0), radius=3)
"""