from functools import partial
from itertools import starmap
from typing import Callable, Iterator, List, Optional, Tuple, Union

import pandas as pd
from democratizing_data_ml_algorithms.data.repository import Repository, SnippetRepositoryMode



class InterleavedSnippetRepository(Repository):
    def __init__(
        self,
        mode: SnippetRepositoryMode,
        repos: Tuple[Repository],
        fractions: Optional[Tuple[float]],
        repeat_longest: Optional[bool] = False,
    ):
        self.mode:SnippetRepositoryMode = mode
        self.repos:Tuple[Repository] = repos
        self.fractions:Tuple[float] = fractions
        self.repeat_longest:bool = repeat_longest

    def get_iter_or_df(
        self,
        getattr_name: str,
        batch_size: Optional[int] = None,
        balance_labels: Optional[bool] = False,
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:

        if batch_size is not None:
            batch_sizes = [int(batch_size * f) for f in self.fractions]
            batch_sizes[-1] += batch_size - sum(batch_sizes)
        else:
            batch_sizes = [None] * len(self.repos)

        def get_with_repeat(
            repo_iter_f:Callable[[], Iterator[pd.DataFrame]],
            repeat:bool,
        ) -> Iterator[Tuple[bool, pd.DataFrame]]:

            has_elapsed = False
            while True:
                for batch in repo_iter_f():
                    yield (has_elapsed, batch)

                has_elapsed = True
                if not repeat:
                    break


        def iter_f():

            all_elapsed = [False] * len(self.repos)

            while True:
                if all(all_elapsed):
                    break

                batches = []
                for i in range(len(self.repos)):
                    if not all_elapsed[i]:
                        elapsed, batch = next(get_with_repeat(
                            partial(
                                getattr(self.repos[i], getattr_name),
                                batch_sizes[i],
                                balance_labels
                            ),
                            self.repeat_longest
                        ))

                        all_elapsed[i] = elapsed

                        batches.append(batch)

                large_batch = pd.concat(batches)

                yield large_batch

        if batch_size:
            return iter_f()
        else:
            return pd.concat(
                list(
                    starmap(
                        lambda repo, batch_size: partial(
                            getattr(repo, getattr_name),
                            batch_size,
                            balance_labels,
                        ),
                        zip(self.repos, batch_sizes),
                    )
                )
            )

    def get_training_data(
        self, batch_size: Optional[int] = None, balance_labels: Optional[bool] = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:

        return self.get_iter_or_df(
            "get_training_data",
            batch_size,
            balance_labels,
        )


    def get_test_data(
            self, batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        return self.get_iter_or_df("get_test_data", batch_size)


    def get_validation_data(
            self, batch_size: Optional[int] = None
    ) -> pd.DataFrame:
        return self.get_iter_or_df("get_validation_data", batch_size)


if __name__ == "__main__":
    from src.data.snippet_repository import SnippetRepository
    from src.data.validated_snippets_repository import ValidatedSnippetsRepository
    mode = SnippetRepositoryMode.MASKED_LM
    repo1 = SnippetRepository(mode)
    repo2 = ValidatedSnippetsRepository(mode)

    repo = InterleavedSnippetRepository(
        mode,
        (repo1, repo2),
        (0.25, 0.75),
        True,
    )

    for batch in repo.get_training_data(10):
        print(batch)
        break


