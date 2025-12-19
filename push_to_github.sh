#!/bin/bash
# Script to push code to GitHub with proper authentication

set -e

echo "=== GitHub Push Script ==="
echo ""

# Ensure we're using HTTPS
git remote set-url origin https://github.com/Arvind-55555/MicroPlastics-Blood-Water-Soil.git

echo "Current remote URL:"
git remote -v
echo ""

# Check if there are changes to commit
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "Staging changes..."
    git add .
    
    echo "Committing changes..."
    git commit -m "Update Microplastic Detection ML Pipeline"
fi

echo ""
echo "Ready to push to GitHub"
echo ""
echo "IMPORTANT: You'll need to authenticate:"
echo "  - Username: Arvind-55555"
echo "  - Password: Use a Personal Access Token (NOT your GitHub password)"
echo ""
echo "To generate a Personal Access Token:"
echo "  1. Go to: https://github.com/settings/tokens"
echo "  2. Click 'Generate new token (classic)'"
echo "  3. Select scope: 'repo' (full control)"
echo "  4. Copy the token and use it as password"
echo ""
read -p "Press Enter to continue with push, or Ctrl+C to cancel..."

# Push to GitHub
echo ""
echo "Pushing to GitHub..."
git push -u origin main

echo ""
echo "✅ Push complete!"
echo ""
echo "Next: Enable GitHub Pages at:"
echo "https://github.com/Arvind-55555/MicroPlastics-Blood-Water-Soil/settings/pages"
echo "  - Source: Deploy from a branch"
echo "  - Branch: main"
echo "  - Folder: /docs"

