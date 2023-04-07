from itertools import starmap
from typing import Iterator, Optional, Tuple, Union

import pandas as pd
from src.data.repository import Repository, SnippetRepositoryMode


class InterleavedSnippetRepository(Repository):
    def __init__(
        self,
        mode: SnippetRepositoryMode,
        repos: Tuple[Repository],
        fractions: Optional[Tuple[float]],
        repeat_longest: Optional[bool] = False,
    ):
        self.mode = mode
        self.repos = repos
        self.fractions = fractions
        self.repeat_longest = repeat_longest

    def get_iter_or_df(
        self, batch_size: Optional[int] = None, balance_labels: Optional[bool] = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:

        batch_sizes = [int(batch_size * f) for f in self.fractions]
        batch_sizes[-1] += batch_size - sum(batch_sizes)

        def get_with_repeat(
            repo:Repository,
            batch_size:int,
            balance_labels:bool,
            repeat:bool,
        ) -> Iterator[Tuple[bool, pd.DataFrame]]:

            has_elapsed = False
            while True:
                for batch in repo.get_training_data(batch_size, balance_labels):
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
                            self.repos[i],
                            batch_sizes[i],
                            balance_labels,
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
                        lambda repo, batch_size: repo.get_training_data(
                            batch_size, balance_labels, balance_labels
                        ),
                        zip(self.repos, batch_sizes),
                    )
                )
            )

    def get_training_data(
        self, batch_size: Optional[int] = None, balance_labels: Optional[bool] = False
    ) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
        def iter_f():
            pass

        if batch_size:
            return iter_f()
        else:
            return pd.DataFrame()
