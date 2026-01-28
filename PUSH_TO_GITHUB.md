# 🚀 Quick Reference - Push to GitHub

## Method 1: Automated Script (Easiest!)

1. Edit `push_to_github.ps1`:
   - Change `YOUR_USERNAME` to your GitHub username
   
2. Run in PowerShell:
   ```powershell
   cd C:\Users\umera\ml-pipeline-ludwig
   .\push_to_github.ps1
   ```

3. Enter GitHub credentials when prompted

Done! ✅

---

## Method 2: Manual Commands

```powershell
# 1. Navigate to project
cd C:\Users\umera\ml-pipeline-ludwig

# 2. Initialize git
git init
git branch -M main

# 3. Add and commit files
git add .
git commit -m "Initial commit: Ludwig ML Pipeline for AIMLFW"

# 4. Add remote (CHANGE YOUR_USERNAME!)
git remote add origin https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git

# 5. Push
git push -u origin main
```

---

## Before Running Commands

### Create GitHub Repository First!

1. Go to: https://github.com/new
2. Repository name: `ludwig-ml-pipeline`
3. Description: `YAML-based ML Pipeline with Ludwig AI`
4. **DO NOT** check any boxes
5. Click "Create repository"

### Get GitHub Token (Not Password!)

1. Go to: https://github.com/settings/tokens
2. Generate new token (classic)
3. Name: "Ludwig ML Pipeline"
4. Select: ☑ repo
5. Generate token
6. **Copy the token!**

---

## When Prompted for Credentials

```
Username: your-github-username
Password: paste-your-token-here
```

**Note**: Password = Personal Access Token, not your GitHub password!

---

## Verification

After push, check:
- https://github.com/YOUR_USERNAME/ludwig-ml-pipeline
- Should see all files ✅
- README.md displayed ✅
- GitHub Actions running ✅

---

## Clone on Linux

```bash
git clone https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git
cd ludwig-ml-pipeline
./install.sh
```

---

## Troubleshooting

**Git not installed?**
```powershell
winget install Git.Git
# Restart PowerShell
```

**Remote already exists?**
```powershell
git remote remove origin
git remote add origin https://github.com/YOUR_USERNAME/ludwig-ml-pipeline.git
```

**Authentication failed?**
- Use Personal Access Token, not password
- Get token: https://github.com/settings/tokens

---

**That's it! Choose Method 1 (script) or Method 2 (manual).** 🎯
