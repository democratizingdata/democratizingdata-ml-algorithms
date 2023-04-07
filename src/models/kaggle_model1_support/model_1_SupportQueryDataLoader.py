import numpy as np

from sklearn.utils import shuffle
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences

WIN_SIZE = 200
SEQUENCE_LENGTH = 320


class SupportQueryDataLoader(Sequence):
    def __init__(
        self,
        data,
        tokenizer,
        training_steps=500,
        batch_size=32,
        is_train=False,
        query_dataloader=None,
        query_masked=False,
        return_query_ids=False,
        return_query_labels=False,
    ):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data = data.fillna("")
        self.is_train = is_train
        self.len = training_steps
        self.query_dataloader = query_dataloader
        self.query_masked = query_masked
        self.return_query_ids = return_query_ids
        self.return_query_labels = return_query_labels

        self.on_epoch_end()

    def _create_group_data(self):
        all_unique_group = list(self.data.group.unique())
        for group in all_unique_group:
            self.data_group[group] = list(
                zip(
                    list(self.data[self.data["group"] == group].title),
                    list(self.data[self.data["group"] == group].text),
                    list(self.data[self.data["group"] == group].label),
                )
            )

        self.all_unique_group = all_unique_group

    def on_epoch_end(self):
        if self.is_train:
            for k in list(self.data_group.keys()):
                self.data_group[k] = shuffle(self.data_group[k])

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # step 1: create support/group data samples
        support_samples = []
        support_labels = []
        support_classes = []
        query_samples = []
        query_labels = []
        query_classes = []
        (
            query_ids,
            query_samples,
            query_labels,
            query_classes,
        ) = self.query_dataloader.__getitem__(index)
        if self.return_query_ids is False:
            query_ids = None

        # step 3: tokenize and return compute sequence label
        query_batch = {}
        query_batch["input_ids"] = []
        query_batch["attention_mask"] = []
        query_batch["token_type_ids"] = []
        query_batch["ids"] = []

        for i in range(len(query_samples)):
            out = self._process_data(
                query_samples[i], query_labels[i], self.query_masked
            )
            query_batch["input_ids"].append(out["input_ids"])
            query_batch["attention_mask"].append(out["attention_mask"])
            query_batch["token_type_ids"].append(out["token_type_ids"])
            if query_ids is not None:
                query_batch["ids"].append(query_ids[i])

        # step 4: padding to max len
        query_batch["input_ids"] = pad_sequences(
            query_batch["input_ids"],
            padding="post",
            value=self.tokenizer.pad_token_id,
        )
        for k in ["attention_mask", "token_type_ids"]:
            pad_value = 0
            query_batch[k] = pad_sequences(
                query_batch[k], padding="post", value=pad_value
            )

        for k in list(["input_ids", "attention_mask", "token_type_ids"]):
            vals = np.array(query_batch[k]).astype(np.int32)
            if len(vals.shape) == 3:
                vals = np.squeeze(vals, axis=0)
            query_batch[k] = vals

        return query_batch

    def _process_data(self, inp_string, label_string, masked_label=False):
        input_tokenize = self.tokenizer(
            inp_string,
            return_offsets_mapping=True,
            max_length=SEQUENCE_LENGTH,
            truncation=True,
        )
        results = {
            "input_ids": input_tokenize["input_ids"],
            "attention_mask": input_tokenize["attention_mask"],
            "token_type_ids": [0] * len(input_tokenize["input_ids"]),
        }
        return results
