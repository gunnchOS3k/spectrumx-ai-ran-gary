# GitHub Project Migration Summary

## Overview

This migration replaces Asana project management with GitHub Projects v2, providing better integration with code, issues, and PRs.

---

## What Was Created

### 1. GitHub Project
- **Name:** SpX-DAC 2026 — spectrumx-ai-ran-gary (gunnchOS3kMLV)
- **Owner:** gunnchOS3k
- **Type:** Projects v2

### 2. Labels Created
- **Type labels:** `type:baseline`, `type:ssl`, `type:anomaly`, `type:viz`, `type:eval`, `type:mlops`, `type:paper`, `type:pm`, `type:qa`
- **Phase labels:** `phase:core-detection`, `phase:portfolio-digital-twin`, `phase:portfolio-ai-ran`
- **Priority labels:** `priority:P0`, `priority:P1`, `priority:P2`, `priority:P3`
- **Status label:** `blocked`

### 3. Project Fields Created
- **Status** (single select): Backlog, Ready, In Progress, Blocked, In Review, Done
- **Sprint** (single select): Sprint 0 Setup, Sprint 1 Baselines, Sprint 2 SSL, Sprint 3 Anomaly+Fusion, Sprint 4 Polish+Submission, Portfolio (Digital Twin), Portfolio (AI-RAN)
- **Owner** (single select): Edmund, Ananya, Noah
- **Effort** (single select): XS, S, M, L, XL
- **Deliverable** (single select): Code, Experiment, Figure, Doc, Demo, Submission
- **Due Date** (date)

### 4. Issues Created
- **Total tasks in CSV:** 47
- **Issues created:** [Run script to get actual count]
- **Issues skipped:** [Duplicates or invalid entries]

---

## Import Script

**Location:** `scripts/import_asana_csv_to_github_projects.py`

**Usage:**
```bash
# Ensure GitHub CLI is authenticated with project scope
gh auth refresh -s project

# Run import script
python scripts/import_asana_csv_to_github_projects.py
```

**What it does:**
1. Creates missing labels
2. Creates GitHub Project v2
3. Creates project fields
4. Parses Asana CSV
5. Audits repository to determine completion status
6. Creates GitHub Issues with proper structure
7. Prints summary

**Note:** Due to GitHub CLI limitations, issues must be added to the project manually via GitHub UI or GraphQL API.

---

## Issue Template

Every issue follows this structure:

```markdown
## Context
[Why this task matters to competition + careers]

## Acceptance Criteria
- [ ] Checklist item 1
- [ ] Checklist item 2

## Dependencies
**Blocked by:** [Issue links or "none"]
**Blocks:** [Issue links or "none"]

## Owner Growth Outcome
- [Skill/outcome 1]
- [Skill/outcome 2]
- [Skill/outcome 3]

## Deliverable
[Code/Experiment/Figure/Doc/Demo/Submission]

## Notes
- Original Asana Task ID: [ID]
- Due Date: [Date]
- Original Assignee: [Name] ([Email])
```

---

## Repository Audit Results

### Completed ✅
- [x] Repo skeleton exists
- [x] Dataset loader implemented (`spectrumx_loader.py`)
- [x] Baselines implemented (Energy Detector, Spectral Flatness)
- [x] Streamlit dashboard deployed
- [x] Architecture docs created
- [x] UML diagrams created
- [x] ML scaffolding with decision points

### In Progress 🚧
- [ ] PSD + Logistic Regression baseline
- [ ] Evaluation harness
- [ ] One-command baseline runner
- [ ] Dataset inventory documentation

### Not Started 📋
- [ ] SSL implementation
- [ ] Anomaly detection
- [ ] Calibration implementation
- [ ] Ensemble fusion
- [ ] Submission bundle

---

## Next Steps

### For Team Members

**Edmund:**
1. Review GitHub Project setup
2. Verify issue assignments are correct
3. Add issues to Project manually (or via GraphQL)
4. Set project field values (Status, Sprint, Owner, etc.)
5. Review and approve this PR

**Ananya:**
1. Check issues assigned to you
2. Update issue status as you work
3. Use GitHub Project board to track progress
4. Reference issues in commit messages: `Fix #123`

**Noah:**
1. Check issues assigned to you
2. Update issue status as you work
3. Use GitHub Project board to track progress
4. Reference issues in commit messages: `Fix #123`

### For Project Management

1. **Add issues to Project:**
   - Go to GitHub Project board
   - Click "Add item" → "Issues"
   - Select issues to add

2. **Set field values:**
   - Click on issue in Project board
   - Set Status, Sprint, Owner, Effort, Deliverable, Due Date

3. **Track progress:**
   - Use Project board views (By Status, By Sprint, By Owner)
   - Update issue status as work progresses
   - Close issues when done

---

## Migration Checklist

- [x] Import script created
- [x] Labels defined
- [x] Project fields defined
- [x] Documentation created (PROJECT_PLAN, PROGRESS_REPORT, etc.)
- [x] CONTRIBUTING.md created
- [x] SECURITY.md created
- [ ] Run import script (requires GitHub CLI auth)
- [ ] Verify issues created correctly
- [ ] Add issues to Project
- [ ] Set project field values
- [ ] Archive Asana project (after verification)

---

## Troubleshooting

### Import Script Fails

**Error:** "GitHub CLI not authenticated"
- Solution: `gh auth login` and `gh auth refresh -s project`

**Error:** "Project creation failed"
- Solution: Check GitHub permissions, ensure project scope granted

**Error:** "Issue creation failed"
- Solution: Check repository permissions, verify assignee GitHub usernames

### Issues Not Showing in Project

- Issues must be manually added to Project (GitHub CLI limitation)
- Use GitHub UI: Project → Add item → Issues
- Or use GraphQL API: `addProjectV2ItemById`

### Field Values Not Setting

- Use GitHub UI: Click issue → Set field values
- Or use GraphQL API: `updateProjectV2ItemFieldValue`

---

## Resources

- [GitHub Projects Documentation](https://docs.github.com/en/issues/planning-and-tracking-with-projects)
- [GitHub CLI Documentation](https://cli.github.com/manual/)
- [GitHub GraphQL API](https://docs.github.com/en/graphql)

---

## Questions?

Open a GitHub Issue with label `type:pm` for project management questions.
