from democratizing_data_ml_algorithms.data.repository import Repository

VALID_REPOS = ["kaggle", "entity", "snippet", "icsr_parquet"]
REPO_HELP_TEXT = (
    f"REPO indicates repository to use, valid options are: {','.join(VALID_REPOS)}"
)


def resolve_repo(repo_name: str) -> Repository:
    if repo_name == "kaggle":
        from democratizing_data_ml_algorithms.data.kaggle_repository import (
            KaggleRepository,
        )

        return KaggleRepository()
    elif repo_name == "entity":
        from democratizing_data_ml_algorithms.data.entity_repository import (
            EntityRepository,
        )

        return EntityRepository()
    elif "snippet" in repo_name:
        from democratizing_data_ml_algorithms.data.snippet_repository import (
            SnippetRepository,
        )

        return SnippetRepository(repo_name.split("-")[1])
    elif "icsr_parquet" in repo_name:
        from democratizing_data_ml_algorithms.data.icsr_parquet_repository import (
            IcsrParquetRepository,
        )

        parquet_file_names = repo_name.split("$")[1].split(",")
        return IcsrParquetRepository(parquet_file_names)
    else:
        raise ValueError(
            f"Unknown repository: {repo_name}. Valid options are: {','.join(VALID_REPOS)}"
        )
