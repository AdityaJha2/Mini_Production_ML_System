# Put this project on GitHub

## 1. Create an empty repository
1. Sign in at https://github.com and click **+ > New repository**.
2. Name it `fraud-detection-ml-system`. Set visibility to **Public**.
3. Do **not** tick "Add a README", ".gitignore" or "license". The project already has them.
4. Click **Create repository** and copy the URL, e.g. `https://github.com/AdityaJha2/Mini_Production_ML_System.git`.

## 2. Push the project from your computer
Install Git if needed (`git --version`). Then, inside the project folder:

```bash
git init
git add .
git commit -m "Card fraud detection: mini production ML system (Option B)"
git branch -M main
git remote add origin https://github.com/AdityaJha2/Mini_Production_ML_System.git
git push -u origin main
```

When asked for a password, use a GitHub Personal Access Token (Settings > Developer settings >
Personal access tokens, scope `repo`), or sign in with the GitHub CLI: `gh auth login`.

## 3. Pull it onto any machine
```bash
git clone https://github.com/AdityaJha2/Mini_Production_ML_System.git
cd fraud-detection-ml-system
make install && make all
```

## 4. Before submitting
The repository is https://github.com/AdityaJha2/Mini_Production_ML_System. Rebuild the report with
`make report`, then commit and push again.
