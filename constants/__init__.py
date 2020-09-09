SEGMENTATION_FILE_NAME = 'segmentation.txt'

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 50
IMAGE_SCALE = 3

SCALED_WIDTH = IMAGE_WIDTH * IMAGE_SCALE
SCALED_HEIGHT = IMAGE_HEIGHT * IMAGE_SCALE

SELECT_X_MAX_DISTANCE = 5
LINE_WIDTH = 2 * IMAGE_SCALE

CRAWLER_HEADERS = {
    'Host': 'zhjwxk.cic.tsinghua.edu.cn',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/85.0.4183.83 Safari/537.36',
}

CLASSES = ['#', '+', '2', '3', '4', '6', '7', '8', '9', 'B', 'C', 'E', 'F',
           'G', 'H', 'J', 'K', 'M', 'P', 'Q', 'R', 'T', 'V', 'W', 'X', 'Y']
CLASSES_TO_ID = {k: i for i, k in enumerate(CLASSES)}
