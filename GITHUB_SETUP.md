# Publishing this repository on GitHub

Everything below runs on your Mac, from the folder that contains `teaching_pipeline/` and
`teaching_pipeline_v2/`.

```bash
cd /Users/sijanregmi/Desktop/Projects/Medical_Physics/Project_1
```

---

## 1. Before the first commit: three edits

`.gitignore`, `LICENSE`, `CITATION.cff` and the workflow are already in place. Three things still carry
placeholders or need a decision from you.

| What | Where | Why |
|---|---|---|
| `USERNAME` → your GitHub username | `README.md` (clone URL), `CITATION.cff` (`repository-code`) | two occurrences; a plain find-and-replace |
| Copyright holder | `LICENSE` line 3, `CITATION.cff` (`authors`) | currently "Sijan Regmi" — correct the spelling or add co-authors |
| License choice | `LICENSE` | MIT is the permissive default: anyone may reuse with attribution. If your institution or Canon Medical has a policy on publishing work derived from this paper, check it before making the repo public. |

```bash
grep -rn "USERNAME" README.md CITATION.cff     # find the placeholders
```

## 2. Check what will be uploaded

The paper PDF and the slide deck are deliberately excluded (`*.pdf`, `*.pptx` in `.gitignore`) because they
are copyright of the paper's authors and publisher. Generated output (`results/`) and the virtual
environments (`.venv/`) are excluded too — `results/` alone was over 50 MB.

```bash
git init
git add -A
git status --short | head -40      # review the list
git status --short | wc -l         # expect roughly 70-80 files, no .pdf, no .pptx, no .venv
```

If anything unwanted appears, add it to `.gitignore` and run `git add -A` again.

## 3. First commit

```bash
git branch -M main
git commit -m "SiPM multiplexing in a monolithic PET detector: teaching re-implementation (v1 CPU, v2 GPU)"
```

If git asks who you are:

```bash
git config --global user.name "Sijan Regmi"
git config --global user.email "you@example.com"
```

## 4. Create the repository and push

**With the GitHub CLI** (`brew install gh` if you do not have it):

```bash
gh auth login
gh repo create sipm-multiplexing-pet --public --source=. --remote=origin --push
```

**Or through the website:** create an empty repository at <https://github.com/new> — no README, no
`.gitignore`, no license, since this repository already has them — then:

```bash
git remote add origin https://github.com/regmi-sijan/sipm-multiplexing-pet.git
git push -u origin main
```

Use `--private` (or the private option on the website) if you want to review it before it is public. You can
switch it to public later in Settings.

## 5. After the first push

- **Check the CI run** under the Actions tab. `.github/workflows/tests.yml` installs each version, runs
  `check_env.py`, runs the test suites, and does a two-epoch smoke run — on Python 3.11 and 3.12, for both
  versions. It takes a few minutes, mostly installing PyTorch.
- **Add a CI badge** to the top of `README.md` once you know the repo URL:
  ```markdown
  [![tests](https://github.com/regmi-sijan/sipm-multiplexing-pet/actions/workflows/tests.yml/badge.svg)](https://github.com/regmi-sijan/sipm-multiplexing-pet/actions/workflows/tests.yml)
  ```
- **Fill in the repo description and topics** (Settings, or the gear icon on the main page):
  suggested topics `pet-imaging`, `sipm`, `medical-physics`, `monte-carlo`, `pytorch`, `cnn`.
- **Confirm the figure renders.** `README.md` shows `docs/fwhm_vs_channels_medium.png`; if it appears broken,
  the file was not committed.

## 6. Everyday use afterwards

```bash
git add -A
git commit -m "what changed"
git push
```

Generated results are ignored, so a new run never shows up as a change. To publish a particular result, copy
it into that version's `example_results/` folder, which is committed on purpose:

```bash
cp teaching_pipeline_v2/results/medium/summary.csv teaching_pipeline_v2/example_results/
```

## Optional: a cleaner history

This first commit contains both versions at once, so the repository will not show v1 and v2 as separate
moments in time. If you would like the history to reflect the actual development, commit v1 first, then v2:

```bash
git init && git branch -M main
git add .gitignore LICENSE teaching_pipeline && git commit -m "v1: CPU teaching pipeline"
git add -A && git commit -m "v2: GPU support, RMS metric, experiment tooling, docs"
```

Tagging the versions also helps anyone browsing later:

```bash
git tag -a v1.0 -m "CPU version" && git tag -a v2.0 -m "GPU version"
git push --tags
```
