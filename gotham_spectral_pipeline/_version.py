import git

__all__ = ["__version__"]

repo = git.Repo(__file__, search_parent_directories=True)
sha = repo.git.rev_parse("--short", "HEAD")
dirty = repo.is_dirty()

__version__ = f"{sha}_{'dirty' if dirty else 'clean'}"
