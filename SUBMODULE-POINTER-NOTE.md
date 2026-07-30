# Preserved submodule pointer

Recorded 2026-07-30, before committing a staged deletion of the `language_models`
gitlink. The repository has no `.gitmodules` file, so once the gitlink is gone from
the tree the commit id below is the only remaining pointer to that submodule's
history.

```
mode   160000 (gitlink)
path   language_models
commit 63b368aaee74ec077abcc8f66f0c8fbd4c712a07
parent commit in this repo  097f496bca630ef242cfe9e6b63bc8b9a420e14d ("Added language_models repository")
```

The submodule URL is UNKNOWN: it was never recorded in `.gitmodules` in any commit
reachable from this branch, and `git config` in this working tree carries no
`submodule.language_models.url` entry either. If the submodule content is needed
again, the commit id above is the identifier to search for; the hosting remote has
to be recovered from outside this repository.

To restore the gitlink without the content:

```
git update-index --add --cacheinfo 160000,63b368aaee74ec077abcc8f66f0c8fbd4c712a07,language_models
```
