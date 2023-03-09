import math

from tensorflow.keras.utils import Sequence

class QueryDataLoader(Sequence):
    def __init__(self, data, batch_size=32):
        self.batch_size = batch_size
        self.data = data.fillna("")
        self.batch_ids = self.data["id"].tolist()
        self.batch_text = self.data["text"].tolist()
        self.batch_label = self.data["label"].tolist()

    def __len__(self):
        return math.ceil(len(self.batch_text) / self.batch_size)

    def __getitem__(self, index):
        id = self.batch_ids[index * self.batch_size : (index + 1) * self.batch_size]
        text = self.batch_text[index * self.batch_size : (index + 1) * self.batch_size]
        label = self.batch_label[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        classes = [1 if l != "" else 0 for l in label]
        return id, text, label, classes
