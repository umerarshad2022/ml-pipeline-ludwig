# 🚀 Creating GitHub Repository - Step-by-Step Guide

## Quick GitHub Setup (5 Minutes)

### Step 1: Create Repository on GitHub

1. Go to: https://github.com/new

2. Fill in details:
   ```
   Repository name:    ludwig-ml-pipeline
   Description:        YAML-based ML Pipeline Engine with Ludwig AI for O-RAN/AIMLFW
   Visibility:         Public (or Private if you prefer)
   Initialize:         ☐ Do NOT check any boxes (we have files already)
   ```

3. Click **"Create repository"**

### Step 2: Push Code from Your Windows Machine

```powershell
# Open PowerShell in: C:\Users\umera\ml-pipeline-ludwig

# Initialize git
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: Ludwig ML Pipeline for AIMLFW"

# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git

# Push to GitHub
git push -u origin main
```

**If `main` branch errors**, try:
```powershell
git branch -M main
git push -u origin main
```

### Step 3: Verify on GitHub

Go to: `https://github.com/YOUR_USERNAME/ludwig-ml-pipeline`

You should see:
- ✅ All files uploaded
- ✅ README.md displayed
- ✅ GitHub Actions ready (in .github/workflows/)

---

## Using GitHub with SSH (Recommended)

### Setup SSH Key

```powershell
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Press Enter for all prompts (use default locations)

# Copy public key to clipboard
type C:\Users\umera\.ssh\id_ed25519.pub | clip
```

### Add to GitHub

1. Go to: https://github.com/settings/keys
2. Click **"New SSH key"**
3. Paste your key
4. Click **"Add SSH key"**

### Use SSH Remote

```powershell
# Change remote to SSH
git remote set-url origin git@github.com:YOUR_USERNAME/ludwig-ml-pipeline.git

# Test connection
ssh -T git@github.com

# Should see: "Hi YOUR_USERNAME! You've successfully authenticated"

# Now you can push without password
git push
```

---

## Clone on Your Linux Test Machine

Once repository is on GitHub:

```bash
# SSH to your Linux machine
ssh user@your-linux-machine

# Clone repository
git clone https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git

# Or with SSH:
git clone git@github.com:YOUR_USERNAME/ludwig-ml-pipeline.git

# Enter directory
cd ludwig-ml-pipeline

# Run installation
chmod +x install.sh
./install.sh
```

---

## Repository Management

### Update Code

#### On Windows:
```powershell
cd C:\Users\umera\ml-pipeline-ludwig

# Make changes to files...

# Commit and push
git add .
git commit -m "Description of changes"
git push
```

#### On Linux:
```bash
cd ~/ludwig-ml-pipeline

# Pull latest changes
git pull

# Or pull and reinstall if dependencies changed
git pull
source venv/bin/activate
pip install -r requirements.txt
```

---

## Directory Structure on GitHub

```
ludwig-ml-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── pipelines/
│   ├── __init__.py
│   ├── engine.py               # Main pipeline engine
│   └── handlers/
│       ├── __init__.py
│       ├── data_loader.py
│       ├── ludwig_handler.py
│       └── ludwig_visualize.py
├── examples/
│   └── du-cell-ludwig.yaml     # Example pipeline
├── data/
│   └── du_cell_report.csv      # Sample data
├── .gitignore                  # Git ignore rules
├── Dockerfile                  # Docker image
├── docker-compose.yml          # Docker services
├── install.sh                  # Linux installation script
├── deploy.sh                   # Linux deployment script
├── requirements.txt            # Python dependencies
├── run_pipeline.py             # CLI runner
├── README.md                   # Main documentation
├── QUICK_START.md              # Quick start guide
├── LINUX_DEPLOYMENT.md         # Linux deployment guide
├── PROJECT_COMPARISON.md       # Comparison with original engine
└── LICENSE                     # MIT License
```

---

## Recommended GitHub Settings

### 1. Branch Protection

1. Go to: Repository → Settings → Branches
2. Add rule for `main` branch:
   - ☑ Require pull request before merging
   - ☑ Require status checks to pass (CI/CD)

### 2. Enable Actions

1. Go to: Repository → Settings → Actions → General
2. Select: "Allow all actions and reusable workflows"

### 3. Add Topics

1. Go to repository main page
2. Click ⚙️ (settings icon) next to "About"
3. Add topics:
   ```
   machine-learning
   deep-learning
   ludwig-ai
   mlops
   yaml-based
   aimlfw
   o-ran
   anomaly-detection
   ```

### 4. Create Releases

```powershell
# Tag a release
git tag -a v1.0.0 -m "Initial release"
git push origin v1.0.0
```

Then on GitHub:
1. Go to: Releases → Create new release
2. Choose tag: v1.0.0
3. Title: "Ludwig ML Pipeline v1.0.0"
4. Description: "Initial release with Ludwig AI integration"
5. Click "Publish release"

---

## Collaborative Workflow

### For Team Members

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git
cd ludwig-ml-pipeline

# 2. Create feature branch
git checkout -b feature/new-model

# 3. Make changes...

# 4. Commit and push
git add .
git commit -m "Add new model support"
git push origin feature/new-model

# 5. Create Pull Request on GitHub
```

### Code Review Process

1. Team member creates Pull Request
2. CI/CD runs automatically
3. Review code changes
4. Merge when approved

---

## Private vs Public Repository

### Public Repository (Recommended for Open Source)

**Pros:**
- ✅ Community contributions
- ✅ Portfolio showcase
- ✅ Free GitHub Actions (2000 min/month)
- ✅ GitHub Pages for documentation

**Cons:**
- ⚠️ Code is visible to everyone
- ⚠️ Need to manage license

### Private Repository

**Pros:**
- ✅ Code is private
- ✅ Controlled access
- ✅ Free for unlimited collaborators

**Cons:**
- ⚠️ Limited Actions minutes (2000 min/month)
- ⚠️ No community contributions

**Recommendation**: Start public for learning, can make private later

---

## License

Add LICENSE file:

```powershell
# Create MIT License
@"
MIT License

Copyright (c) 2026 YOUR_NAME

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"@ | Out-File -FilePath LICENSE -Encoding utf8
```

Then commit:
```powershell
git add LICENSE
git commit -m "Add MIT License"
git push
```

---

## Troubleshooting

### Issue: Authentication Failed

```powershell
# Use Personal Access Token instead of password
# 1. Generate token: https://github.com/settings/tokens
# 2. Use token as password when pushing
```

### Issue: Large Files

```powershell
# Check file sizes
git ls-files | xargs ls -lh | sort -k 5 -hr | head -10

# If files > 100 MB, use Git LFS
git lfs install
git lfs track "*.pkl"
git lfs track "*.h5"
git add .gitattributes
git commit -m "Add Git LFS"
```

### Issue: Wrong Branch

```powershell
# Rename branch to main
git branch -M main
git push -u origin main
```

---

## Quick Command Reference

```powershell
# Clone repository
git clone https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git

# Check status
git status

# Add files
git add .
git add specific_file.py

# Commit
git commit -m "Your message"

# Push
git push

# Pull latest
git pull

# Create branch
git checkout -b feature/new-feature

# Switch branch
git checkout main

# Merge branch
git merge feature/new-feature

# View history
git log --oneline

# Undo last commit (keep changes)
git reset --soft HEAD~1

# Discard local changes
git checkout -- filename
```

---

## Next Steps

1. **Create Repository**: https://github.com/new

2. **Push Code**:
   ```powershell
   cd C:\Users\umera\ml-pipeline-ludwig
   git init
   git add .
   git commit -m "Initial commit: Ludwig ML Pipeline"
   git remote add origin https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git
   git push -u origin main
   ```

3. **Clone on Linux**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git
   cd ludwig-ml-pipeline
   ./install.sh
   ```

4. **Start Using**:
   ```bash
   source venv/bin/activate
   python run_pipeline.py examples/du-cell-ludwig.yaml
   ```

---

**Repository will be ready for your Linux test machine in 5 minutes!** 🚀
