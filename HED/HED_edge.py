import cv2 as cv
import numpy as np

width = 256
height = 256
# ! [CropLayenr]
class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0
    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]
        # self.ystart = (inputShape[2] - targetShape[2]) / 2
        # self.xstart = (inputShape[3] - targetShape[3]) / 2
        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width
        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:, :, self.ystart:self.yend, self.xstart:self.xend]]
# ! [CropLayer]
# ! [Register]重载这个类，注册该层
cv.dnn_registerLayer('Crop', CropLayer)
# ! [Register]
# Load the model.
path_prototxt = "./HED/deploy.prototxt"
path_caffemodel = "./HED/hed_pretrained_bsds.caffemodel"
# 准备构建网络图并加载权重
net = cv.dnn.readNet(path_prototxt,path_caffemodel)

def Get_edgeMap(input):
    input = np.stack([input,input,input],axis=-1)
    # 批量加载图像，并通过网络运行它们
    # framez:输入图像3通道；scalefactor：图像放缩函数，像素值通常在0-255，默认为1不放缩；
    # size：输出图像的空间大小。它将等于后续神经网络作为blobFromImage输出所需的输入大小；
    # swapRB：布尔值，表示我们是否想在3通道图像中交换第一个和最后一个通道。OpenCV默认图像为BGR格式，但如果我们想将此顺序转换为RGB，我们可以将此标志设置为True，这也是默认值。
    # crop：布尔标志，表示我们是否想居中裁剪图像。如果设置为True，则从中心裁剪输入图像时，较小的尺寸等于相应的尺寸，而其他尺寸等于或大于该尺寸。然而，如果我们将其设置为False，它将保留长宽比，只是将其调整为固定尺寸大小。
    # mean：为了进行归一化，有时我们计算训练数据集上的平均像素值，并在训练过程中从每幅图像中减去它。如果我们在训练中做均值减法，那么我们必须在推理中应用它。这个平均值是一个对应于R, G, B通道的元组。例如Imagenet数据集的均值是R=103.93, G=116.77, B=123.68。如果我们使用swapRB=False，那么这个顺序将是(B, G, R)。
    inp = cv.dnn.blobFromImage(input, scalefactor=1.0, size=(width,height),
                               mean=(104.00698793, 116.66876762, 122.67891434),
                               swapRB=False, crop=False)
    net.setInput(inp)
    out = net.forward()
    out = out[0, 0]
    out = np.expand_dims(out,axis=0)
    return out