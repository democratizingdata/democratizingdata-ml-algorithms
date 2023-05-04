from src.data.repository import Repository, SnippetRepositoryMode

VALID_REPOS = [
    "kaggle",
    "entity",
    *[f"snippet-{s.value}" for s in SnippetRepositoryMode],
]
REPO_HELP_TEXT = (
    f"REPO indicates repository to use, valid options are: {','.join(VALID_REPOS)}"
)


def resolve_repo(repo_name: str) -> Repository:
    if repo_name == "kaggle":
        from src.data.kaggle_repository import KaggleRepository

        return KaggleRepository()
    elif repo_name == "entity":
        from src.data.entity_repository import EntityRepository

        return EntityRepository()
    elif "snippet" in repo_name:
        from src.data.snippet_repository import SnippetRepository

        return SnippetRepository(repo_name.split("-")[1])
    elif "validated" in repo_name:
        from src.data.validated_snippets_repository import ValidatedSnippetsRepository

        return ValidatedSnippetsRepository(repo_name.split("-")[1])
    else:
        raise ValueError(
            f"Unknown repository: {repo_name}. Valid options are: {','.join(VALID_REPOS)}"
        )
