from easydict import EasyDict as edict


__C = edict()

cfg = __C

__C.DATA = edict()
__C.DATA.PC_DATA_PATH = 'E:/Programs/Vis-Inf_Fuse/Dataset'
__C.DATA.SERVER_DATA_PATH = '/DCFDFuse/Dataset'
__C.DATA.IS_RGB = False
__C.DATA.DATA_NAME = 'MSRS' # M3FD, MSRS, RoadScene, TNO
__C.DATA.DATA_SIZE = (240, 320)

__C.TRAIN = edict()
__C.TRAIN.BATCHSIZE = 4
__C.TRAIN.NUM_EPOCHS = 100
__C.TRAIN.EPOCH_GAP = 0  # smaller than num_epochs
__C.TRAIN.PC_LOG_DIR = '/log'
__C.TRAIN.SERCER_LOG_DIR = '/DCFDFuse/log'

__C.TEST = edict()
__C.TEST.BATCHSIZE = 16
__C.TEST.PC_LOG_DIR = './test_log'
__C.TEST.SERCER_LOG_DIR = '/DCFDFused/test_log'