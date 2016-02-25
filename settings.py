class Settings(object):
    def __init__(self, filename):
        self.filename = filename
        self.values = {}
    def load(self):
        try:
            f = open(self.filename)
            for line in f:
                line = line.strip()
                if line.startswith('#'):
                    continue
                pos = line.find('=')
                if pos < 0:
                    continue
                name = line[:pos].strip()
                value = line[pos + 1:].strip()
                self.values[name] = value
            f.close()
        except FileNotFoundError as e:
            pass
    def get(self, name):
        return self.values.get(name)
    def set(self, name, value):
        self.values[name] = value

