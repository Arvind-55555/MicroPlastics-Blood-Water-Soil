# Git Authentication Guide - Fixing SSH/HTTPS Issues

## Problem: Permission Denied (publickey)

This error occurs when:
- SSH keys from a previous system are not available
- GitHub doesn't recognize your current system's SSH key
- SSH authentication is not properly configured

## Solutions

### Solution 1: Use HTTPS (Easiest - Recommended)

**Switch remote URL to HTTPS:**
```bash
git remote set-url origin https://github.com/Arvind-55555/MicroPlastics-Blood-Water-Soil.git
```

**Push using HTTPS:**
- First push: You'll be prompted for username and password
- For password: Use a **Personal Access Token** (not your GitHub password)
  - Generate token: https://github.com/settings/tokens
  - Select scopes: `repo` (full control of private repositories)
  - Copy token and use it as password when prompted

**Store credentials (optional):**
```bash
# Store credentials in git credential helper
git config --global credential.helper store
# Or use cache (credentials stored for 15 minutes)
git config --global credential.helper cache
```

### Solution 2: Generate New SSH Keys

**1. Generate new SSH key:**
```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
# Press Enter to accept default location
# Optionally set a passphrase
```

**2. Add SSH key to ssh-agent:**
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

**3. Copy public key:**
```bash
cat ~/.ssh/id_ed25519.pub
# Copy the entire output
```

**4. Add to GitHub:**
- Go to: https://github.com/settings/keys
- Click "New SSH key"
- Paste the public key
- Click "Add SSH key"

**5. Test connection:**
```bash
ssh -T git@github.com
# Should see: "Hi Arvind-55555! You've successfully authenticated..."
```

**6. Switch back to SSH (if desired):**
```bash
git remote set-url origin git@github.com:Arvind-55555/MicroPlastics-Blood-Water-Soil.git
```

### Solution 3: Use GitHub CLI

**Install GitHub CLI:**
```bash
sudo apt install gh
```

**Authenticate:**
```bash
gh auth login
# Follow prompts to authenticate
```

**Push:**
```bash
git push -u origin main
```

## Quick Fix Commands

**Switch to HTTPS (immediate fix):**
```bash
git remote set-url origin https://github.com/Arvind-55555/MicroPlastics-Blood-Water-Soil.git
git push -u origin main
```

**Check current remote:**
```bash
git remote -v
```

**Reset remote if needed:**
```bash
git remote remove origin
git remote add origin https://github.com/Arvind-55555/MicroPlastics-Blood-Water-Soil.git
```

## Preventing Future Issues

1. **Use HTTPS with credential helper:**
   ```bash
   git config --global credential.helper store
   ```

2. **Or use SSH with proper key management:**
   - Keep SSH keys backed up
   - Use SSH key passphrases
   - Add keys to GitHub when setting up new systems

3. **Use Personal Access Tokens for HTTPS:**
   - More secure than passwords
   - Can be revoked if compromised
   - Generate at: https://github.com/settings/tokens

## Current Setup

Your repository is now configured to use HTTPS:
```
origin  https://github.com/Arvind-55555/MicroPlastics-Blood-Water-Soil.git
```

When you push, you'll be prompted for:
- Username: `Arvind-55555`
- Password: Your Personal Access Token (not GitHub password)

