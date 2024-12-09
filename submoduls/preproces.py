import cv2 


class ClaheProcess:
    def __init__(self, clipLimit = 1, tileGridSize = (4, 4)):
        self.clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(4,4))

    def process(self, image):
        return self.clahe.apply(image)

class GrayScaleProcess:
    def process(image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

class GaussianProcess:
    def __init__(self, kernal = (3, 3), diviation = 0):
        self.kernal = kernal
        self.diviation = diviation

    def process(self, image):
        return cv2.GaussianBlur(image, self.kernal, self.diviation)


class MultiProcess:
    def __init__(self, subprocesses = [GrayScaleProcess,GaussianProcess(),ClaheProcess()]):
        self.subprocesses = subprocesses

    def process(self, image):
        img = image
        for i in self.subprocesses:
            img = i.process(img)
        
        return img

class NoProcess:
    def __init__(self):
        pass

    def process(self, image):
        return image



