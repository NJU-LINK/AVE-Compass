# AVE-Compass Website Release

The publish-ready GitHub Pages site lives in `docs/`.

## Directory

- `docs/index.html`: complete project page, styles, and interactions
- `docs/assets/figures/`: figures used by the page
- `docs/assets/cases/`: five source videos and six model results per case
- `docs/.nojekyll`: disables Jekyll processing for static assets

## Local Preview

```bash
python3 -m http.server 8087 --directory docs
```

Open `http://localhost:8087/`.

## GitHub Pages

1. Upload the repository to GitHub.
2. In **Settings > Pages**, select **Deploy from a branch**.
3. Select the target branch and the `/docs` folder.
4. Save and wait for the Pages deployment to finish.

## Preflight Links

- Dataset: `https://huggingface.co/datasets/NJU-LINK/AVE-Compass`
- Repository: `https://github.com/NJU-LINK/AVE-Compass`

The repository URL currently returns `404` before upload. Verify that it becomes public and reachable after the repository is created.

## Paper Link

After the arXiv page is available, update the centralized setting near the end of `docs/index.html`:

```js
const PROJECT_LINKS = {
  paper: "https://arxiv.org/abs/XXXX.XXXXX",
};
```

Both Paper buttons will use the new URL automatically.
