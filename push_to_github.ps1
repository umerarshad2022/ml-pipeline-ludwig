# Push Ludwig ML Pipeline to GitHub
# ⚠️ REPLACE 'YOUR_USERNAME' WITH YOUR ACTUAL GITHUB USERNAME BEFORE RUNNING!

$GITHUB_USERNAME = "YOUR_USERNAME"  # ← CHANGE THIS!
$REPO_NAME = "ludwig-ml-pipeline"

Write-Host "========================================"
Write-Host "  Pushing Ludwig ML Pipeline to GitHub"
Write-Host "========================================"
Write-Host ""

# Check if username was changed
if ($GITHUB_USERNAME -eq "YOUR_USERNAME") {
    Write-Host "❌ ERROR: Please edit this script and change YOUR_USERNAME to your actual GitHub username!"
    Write-Host ""
    Write-Host "Edit line 4 of this script:"
    Write-Host '$GITHUB_USERNAME = "your-actual-username"'
    exit 1
}

# Navigate to project directory
Write-Host "📁 Navigating to project directory..."
Set-Location C:\Users\umera\ml-pipeline-ludwig

# Initialize git
Write-Host "🔧 Initializing git repository..."
git init
git branch -M main

# Add all files
Write-Host "➕ Adding files..."
git add .

# Show status
Write-Host ""
Write-Host "📊 Files to be committed:"
git status --short

# Commit
Write-Host ""
Write-Host "💾 Creating commit..."
git commit -m "Initial commit: Ludwig ML Pipeline for AIMLFW"

# Add remote
Write-Host ""
Write-Host "🌐 Adding GitHub remote..."
$REMOTE_URL = "https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
Write-Host "   URL: $REMOTE_URL"

git remote remove origin 2>$null  # Remove if exists
git remote add origin $REMOTE_URL

# Push
Write-Host ""
Write-Host "🚀 Pushing to GitHub..."
Write-Host "   (You may be prompted for GitHub credentials)"
Write-Host ""
git push -u origin main

# Verify
if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================"
    Write-Host "  ✅ SUCCESS! Code pushed to GitHub"
    Write-Host "========================================"
    Write-Host ""
    Write-Host "View your repository:"
    Write-Host "https://github.com/$GITHUB_USERNAME/$REPO_NAME"
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. Go to your GitHub repository"
    Write-Host "2. Verify all files are uploaded"
    Write-Host "3. Clone on Linux: git clone https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "❌ Push failed. Common issues:"
    Write-Host "1. Need to create repository on GitHub first: https://github.com/new"
    Write-Host "2. Need Personal Access Token (not password): https://github.com/settings/tokens"
    Write-Host "3. Check username is correct: $GITHUB_USERNAME"
    Write-Host ""
}
