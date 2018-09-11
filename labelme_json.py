import os
import json
import base64

class LabelmeJSON():
    KEY_FLAGS = 'flags'
    KEY_SHAPES = 'shapes'
    KEY_L_COLOR = 'lineColor'
    KEY_F_COLOR = 'fillColor'
    KEY_IMG_PATH = 'imagePath'
    KEY_IMG = 'imageData'

    DEFAULT_L_COLOR = [255, 0, 255, 128]
    DEFAULT_F_COLOR = [85, 170, 127, 128]


    def __init__(self, img_path, flags=None, shapes=None, lineColor=None, fillColor=None):

        self.__img_path = img_path

        self.__json = self.__get_empty_json()

        if flags is not None:
            self.__json[self.KEY_FLAGS] = flags
        if shapes is not None:
            self.__json[self.KEY_SHAPES] = [shape.get() for shape in shapes]
        if lineColor is not None:
            self.__json[self.KEY_L_COLOR] = lineColor
        if fillColor is not None:
            self.__json[self.KEY_F_COLOR] = fillColor
        if os.path.exists(img_path):
            self.__json[self.KEY_IMG_PATH] = os.path.basename(img_path)
            with open(self.__img_path, 'rb') as f:
                #img_base64 = base64.encodestring(f.read())
                img_base64 = base64.encodebytes(f.read())
            self.__json[self.KEY_IMG] = img_base64.decode('utf8')


    def __get_empty_json(self):
        j = {}
        j[self.KEY_IMG_PATH] = ''
        j[self.KEY_IMG] = ''
        j[self.KEY_FLAGS] = {}
        j[self.KEY_L_COLOR] = self.DEFAULT_L_COLOR
        j[self.KEY_F_COLOR] = self.DEFAULT_F_COLOR
        j[self.KEY_SHAPES] = []
        return j


    def get(self):
        return self.__json


    def dumps(self):
        return json.dumps(self.get())


    def dump(self, file_path=None):
        if file_path is None:
            dir_path = os.path.dirname(self.__img_path)
            file_name, _ = os.path.splitext(os.path.basename(self.__img_path))
            file_path = os.path.join(dir_path, file_name + '.json')
        if os.path.exists(file_path):
            os.remove(file_path)
        with open(file_path, 'w') as f:
            json.dump(self.get(), f, indent=2)
        return file_path


class LabelmeJSONShape():
    KEY_LABEL = 'label'
    KEY_L_COLOR = 'line_color'
    KEY_F_COLOR = 'fill_color'
    KEY_POINTS = 'points'

    def __init__(self, label, line_color=None, fill_color=None, points=None):
        self.__shape = {}
        self.__shape[self.KEY_LABEL] = label
        self.__shape[self.KEY_L_COLOR] = line_color
        self.__shape[self.KEY_F_COLOR] = fill_color
        self.__shape[self.KEY_POINTS] = []
        if points is not None:
            self.__shape[self.KEY_POINTS] = points


    def get(self):
        return self.__shape
