#!/bin/bash
# Script to deploy GitHub Pages

set -e

echo "Setting up GitHub Pages deployment..."

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "ERROR: git is not installed. Please install it first:"
    echo "  sudo apt install git"
    exit 1
fi

# Initialize git if not already initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add remote if not exists
if ! git remote | grep -q origin; then
    echo "Adding remote repository..."
    git remote add origin git@github.com:Arvind-55555/MicroPlastics-Blood-Water-Soil.git
else
    echo "Updating remote repository..."
    git remote set-url origin git@github.com:Arvind-55555/MicroPlastics-Blood-Water-Soil.git
fi

# Add all files
echo "Adding files..."
git add .

# Commit if there are changes
if ! git diff --cached --quiet; then
    echo "Committing changes..."
    git commit -m "Deploy GitHub Pages dashboard"
else
    echo "No changes to commit."
fi

# Push to main branch
echo "Pushing to repository..."
git branch -M main
git push -u origin main

echo ""
echo "✅ Deployment complete!"
echo "GitHub Pages will be available at: https://arvind-55555.github.io/MicroPlastics-Blood-Water-Soil"
echo ""
echo "Note: Make sure to enable GitHub Pages in repository settings:"
echo "  Settings → Pages → Source: Deploy from a branch → Branch: main → Folder: /docs"

