from typing import Any, Dict, List

import spacy
from democratizing_data_ml_algorithms.models.text_segmentizer_protocol import (
    TextSegmentizer,
)


class DefaultSpacySegmentizer(TextSegmentizer):
    def __init__(self, config: Dict[str, Any] = dict()):
        self.nlp = spacy.load(config.get("nlp", "en_core_web_sm"))
        self.max_tokens = config.get("max_tokens", 10000)

    def __call__(self, text: str) -> List[str]:

        tokens = text.split()
        if len(tokens) > self.max_tokens:
            texts = [
                " ".join(tokens[i : i + self.max_tokens])
                for i in range(0, len(tokens), self.max_tokens)
            ]
            tokens = tokens[: self.max_tokens]
        else:
            texts = [text]

        process_generator = self.nlp.pipe(
            texts,
            disable=["lemmatizer", "ner", "textcat"],
        )

        sents = []
        for doc in process_generator:
            sents.extend([sent.text for sent in doc.sents])

        return sents
