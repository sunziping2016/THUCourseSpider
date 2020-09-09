import hashlib


class CaptchaCache:
    def __init__(self, filename):
        self.filename = filename
        self.cache = {}

    def load(self):
        with open(self.filename) as f:
            cache = {}
            f = open(self.filename)
            for line in f:
                md5sum, code = line.strip().split('\t')
                cache[md5sum] = code
            self.cache = cache

    def save(self):
        f = open(self.filename, 'w')
        for item in self.cache.items():
            f.write(item[0] + '\t' + item[1] + '\n')

    def search(self, image):
        m = hashlib.md5()
        m.update(image)
        key = m.hexdigest()
        return self.cache.get(key)

    def add(self, image, code):
        m = hashlib.md5()
        m.update(image)
        key = m.hexdigest()
        self.cache[key] = code

    def merge(self, other):
        self.cache.update(other.cache)

    def __len__(self):
        return len(self.cache)
