# Upload to GitHub (manual)

Target: **public** repository  
`https://github.com/ch-auf-der-horst/heine-shannon-internationalitaet`

`gh` may not be installed. Upload via the website + Git:

## 1. Create an empty repository on GitHub

1. Open https://github.com/new (signed in as **ch-auf-der-horst**)
2. Repository name: `heine-shannon-internationalitaet`
3. Public
4. Do **not** add README / .gitignore / License (already present locally)
5. Create repository

## 2. Push

From this repository root:

```bash
git push -u origin main
```

Use browser login, SSH, or a personal access token if GitHub asks for authentication.

## 3. Optional release

GitHub → Releases → tag `v1.0` → optional Zenodo DOI for the paper.
