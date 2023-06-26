from typing import Any, Dict, List, Protocol


class TextSegmentizer(Protocol):
    def __init__(self, config: Dict[str, Any] = dict()):
        """Initializes the segmentizer.

        Args:
            config (Dict[str, Any]): Configuration dictionary.
        """
        ...

    def __call__(self, text: str) -> List[str]:
        """Segmentizes text into a list of segments.

        Args:
            text (str): Text to segmentize.

        Returns:
            List[str]: List of segments.
        """
        ...
