# Security Policy

## Supported Versions

We actively support the current version of the codebase. Security updates will be applied as needed.

## Reporting a Vulnerability

If you discover a security vulnerability, please **do not** open a public issue. Instead:

1. Email the team lead: egunnjr@gunnchos.com
2. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will respond within 48 hours and work with you to resolve the issue.

---

## Security Best Practices

### No Secrets in Repository

**Critical:** Never commit secrets, API keys, tokens, or credentials to the repository.

#### What NOT to Commit:
- ❌ API keys or tokens
- ❌ Passwords or credentials
- ❌ `.env` files with secrets
- ❌ Hardcoded tokens in code
- ❌ Notebooks with exposed tokens
- ❌ Configuration files with secrets

#### What TO Do:
- ✅ Use `.env` files (gitignored) for secrets
- ✅ Use environment variables
- ✅ Use `.env.example` as a template (without secrets)
- ✅ Load secrets at runtime: `os.getenv("SECRET_NAME")`
- ✅ Use GitHub Secrets for CI/CD (if applicable)

### Current Secret Management

#### SpectrumX SDS Token
- **Location:** `.env` file (gitignored)
- **Variable:** `SDS_SECRET_TOKEN`
- **Usage:** Dataset download via `spectrumx_loader.py`
- **Rotation:** Rotate if accidentally exposed

#### Streamlit Secrets (if needed)
- **Location:** `.streamlit/secrets.toml` (gitignored)
- **Usage:** Streamlit Cloud deployment
- **Setup:** Configure via Streamlit Cloud UI

### Verification Checklist

Before committing, verify:

- [ ] No `.env` files staged
- [ ] No hardcoded tokens in code
- [ ] No tokens in notebooks
- [ ] `.gitignore` includes `.env` and `.streamlit/secrets.toml`
- [ ] `git diff` shows no secrets

### If You Accidentally Commit a Secret

**Immediate Actions:**

1. **Rotate the secret immediately**
   - Generate new token/API key
   - Update `.env` file
   - Update any services using the secret

2. **Remove from Git history**
   ```bash
   # Remove file from history (use with caution)
   git filter-branch --force --index-filter \
     "git rm --cached --ignore-unmatch .env" \
     --prune-empty --tag-name-filter cat -- --all
   
   # Force push (coordinate with team first!)
   git push origin --force --all
   ```

3. **Open a Security Issue**
   - Create issue: "SECURITY: Remove token + rotate"
   - Document what was exposed
   - Document rotation steps taken
   - Mark as resolved after cleanup

4. **Notify Team**
   - Inform team members immediately
   - Check if secret was used elsewhere
   - Update documentation if needed

---

## Environment Variables

### Required Variables

Create `.env` file (copy from `.env.example`):

```bash
# SpectrumX SDS API Token
SDS_SECRET_TOKEN=your_token_here
```

### Loading Variables

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Loads .env file

token = os.getenv("SDS_SECRET_TOKEN")
if token is None:
    raise RuntimeError("SDS_SECRET_TOKEN not found in .env")
```

---

## .gitignore Configuration

Our `.gitignore` includes:

```
# Environment files
.env
.env.local
.env.*.local

# Streamlit secrets
.streamlit/secrets.toml

# Data files
*.npy
competition_dataset/
data/raw/
data/processed/

# Results and logs
results/
logs/
```

**Verify:** Run `git check-ignore .env` to confirm `.env` is ignored.

---

## Code Review Checklist

When reviewing PRs, check:

- [ ] No secrets in code
- [ ] No `.env` files added
- [ ] Environment variables used correctly
- [ ] No tokens in commit messages
- [ ] No secrets in documentation

---

## Dataset Security

### Dataset Access

- Dataset requires SpectrumX SDS API token
- Token stored in `.env` (gitignored)
- Dataset files (`.npy`) are gitignored
- No dataset files committed to repository

### Dataset Sharing

- **Internal:** Share via secure channels (not GitHub)
- **External:** Only share with authorized parties
- **Public:** Never share competition dataset publicly

---

## CI/CD Security (if applicable)

If using GitHub Actions or similar:

- Use GitHub Secrets for sensitive values
- Never echo secrets in logs
- Use `${{ secrets.SECRET_NAME }}` syntax
- Rotate secrets regularly

---

## Compliance

### Competition Requirements

- Follow competition data usage policies
- Respect dataset licensing terms
- Do not share competition data publicly

### Academic Integrity

- No plagiarism
- Proper attribution of code/ideas
- Honest reporting of results

---

## Resources

- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [OWASP Top 10](https://owasp.org/www-project-top-ten/)
- [Python Security](https://python.readthedocs.io/en/latest/library/security.html)

---

## Contact

For security concerns, contact:
- **Team Lead:** Edmund Gunn, Jr. (egunnjr@gunnchos.com)
- **GitHub Issues:** Use "SECURITY:" prefix in issue title

---

**Remember:** Security is everyone's responsibility. When in doubt, ask before committing sensitive information.
