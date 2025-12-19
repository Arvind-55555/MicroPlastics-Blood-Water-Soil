#!/bin/bash
# Complete setup and deployment script for GitHub Pages

set -e

echo "=== GitHub Pages Deployment Setup ==="
echo ""

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "Installing git..."
    sudo apt update && sudo apt install -y git
fi

# Initialize git if not already
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Configure git user if not set
if [ -z "$(git config user.name)" ]; then
    echo "Configuring git user..."
    git config user.name "Arvind-55555"
    git config user.email "your-email@example.com"
fi

# Add remote
if ! git remote | grep -q origin; then
    echo "Adding remote repository..."
    git remote add origin git@github.com:Arvind-55555/MicroPlastics-Blood-Water-Soil.git
else
    echo "Updating remote repository..."
    git remote set-url origin git@github.com:Arvind-55555/MicroPlastics-Blood-Water-Soil.git
fi

# Add all files
echo "Adding files to git..."
git add .

# Commit
echo "Committing changes..."
git commit -m "Initial commit: Microplastic Detection ML Pipeline with GitHub Pages dashboard" || echo "No changes to commit"

# Set main branch and push
echo "Pushing to repository..."
git branch -M main
git push -u origin main --force || echo "Push failed - may need to set up SSH keys or use HTTPS"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Go to: https://github.com/Arvind-55555/MicroPlastics-Blood-Water-Soil/settings/pages"
echo "2. Under 'Source', select 'Deploy from a branch'"
echo "3. Select branch: 'main'"
echo "4. Select folder: '/docs'"
echo "5. Click Save"
echo ""
echo "Your dashboard will be available at:"
echo "https://arvind-55555.github.io/MicroPlastics-Blood-Water-Soil"
