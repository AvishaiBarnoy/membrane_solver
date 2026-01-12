# CR Flow Checklist

Use this quick checklist before asking for code review.

## 1) Before Committing

```bash
git status
git diff
git add -p
git diff --staged
```

## 2) After Commits

```bash
git diff main...
git difftool main...
git log --oneline main..HEAD
git log -p main..HEAD
```
