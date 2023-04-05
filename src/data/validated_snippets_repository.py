
from enum import Enum
from src.data.repository import Repository


class SnippetRepositoryMode(Enum):
    NER = "ner"
    CLASSIFICATION = "classification"
    MASKED_LM = "masked_lm"


class ValidatedSnippetsRepository(Repository):
