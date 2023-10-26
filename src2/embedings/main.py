import numpy as np
from sys import argv
import cv2


def optimalSquare(x: int):
    return int(np.ceil(x**0.5)), int(np.ceil(x**0.5))


def closetSquared(x: int):
    return int(np.ceil(x**0.5))**2

def embToIimage(embImg, size=(500,500)):
    embImg = (embImg-embImg.min())/(embImg.max()-embImg.min())*255
    embImg = embImg.astype(np.uint8)
    scale0 = size[0]//embImg.shape[0]
    scale1 = size[1]//embImg.shape[1]
    embImg = np.repeat(np.repeat(embImg, scale0, axis=0), scale1, axis=1)
    #  embImg = cv2.resize(embImg, size, cv2.INTER_NEAREST_EXACT)
    return embImg

emb = np.loadtxt(argv[1])
emb = np.resize(emb, (closetSquared(emb.shape[0])))
#  emb = emb.resize((closetSquared(emb.shape[0])))
embs = np.loadtxt(argv[1]+"s")
embs = np.resize(embs, (embs.shape[0], closetSquared(emb.shape[0])))
#  embs.shape[1]
embGTImg = emb.reshape(optimalSquare(emb.shape[0]))
#  embImg = embs[2].reshape((int(emb.shape[0]**0.5), int(emb.shape[0]**0.5)))
embImg = embGTImg + (np.random.rand(*embGTImg.shape)-0.5)/10

embGTImg = embToIimage(embGTImg)
embImg = embToIimage(embImg)

cv2.imwrite(argv[1]+"GT.png", embGTImg)
cv2.imwrite(argv[1]+".png", embImg)



