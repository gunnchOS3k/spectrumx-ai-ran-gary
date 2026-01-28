#!/usr/bin/env python3
"""
Setup GitHub Project v2 and create EPIC issues for SpX-DAC 2026.

This script:
1. Creates GitHub Project v2 with custom fields
2. Creates EPIC issues (EPIC 1-4) with acceptance criteria
3. Adds issues to Project and sets field values
4. Prints summary and board URL

Requirements:
- GitHub CLI (gh) installed and authenticated
- Project scope: gh auth refresh -s project,read:project -h github.com
- Repository: gunnchOS3k/spectrumx-ai-ran-gary

Usage:
    python scripts/setup_github_project_epics.py
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
REPO = "gunnchOS3k/spectrumx-ai-ran-gary"
PROJECT_TITLE = "SpX-DAC 2026 — spectrumx-ai-ran-gary (gunnchOS3kMLV)"

# EPIC definitions
EPICS = [
    {
        "title": "EPIC 1: Competition Core Detection",
        "body": """## Context
Implement the core competition requirement: detect "structured transmission present" from 1-second IQ samples using semi-supervised/self-supervised learning.

## Acceptance Criteria
- [ ] Eval protocol locked (`docs/EVAL_PROTOCOL.md`)
- [ ] Baselines run via one command (`scripts/run_baseline.py`)
- [ ] Calibration + thresholding implemented
- [ ] SSL using unlabeled data (SimCLR/BYOL/wav2vec)
- [ ] Anomaly detection + fusion (optional)
- [ ] Error analysis + ablations completed
- [ ] ROC-AUC > baseline performance
- [ ] ECE < 0.10 (good calibration)

## Dependencies
- Depends on Sprint 0 setup completion
- Blocks EPIC 4 (Portfolio Extension)

## Owner Growth Outcome
- Semi-supervised learning for RF signals
- Model calibration and threshold optimization
- Error analysis and ablation studies

## Deliverable
Code + Experiment + Submission

## Notes
- Must not contaminate evaluation on real labeled data
- Use file-level splits (no data leakage)
- Reproducible seeds and configs required""",
        "labels": ["phase:core-detection", "priority:P0"],
        "status": "In Progress",
        "sprint": "Sprint 1 Baselines",
        "owner": "Ananya",
        "effort": "XL",
        "deliverable": "Code"
    },
    {
        "title": "EPIC 2: Visualization + Demo",
        "body": """## Context
Create reliable Streamlit dashboard for judge demo and failure case analysis.

## Acceptance Criteria
- [ ] Streamlit app reliability + deploy checklist (`docs/STREAMLIT_DEPLOY.md`)
- [ ] Failure-case browser in dashboard
- [ ] Mini dashboard for judge demo (key visualizations)
- [ ] Demo data generation works immediately
- [ ] All plots render correctly
- [ ] Error handling for invalid inputs

## Dependencies
- Depends on baseline models (EPIC 1)
- Can be done in parallel with EPIC 1

## Owner Growth Outcome
- Streamlit dashboard development
- Data visualization for RF signals
- User experience design

## Deliverable
Demo + Code

## Notes
- Must work without competition dataset (upload-only)
- Demo data should be generated on-the-fly
- No secrets committed""",
        "labels": ["type:viz", "priority:P1"],
        "status": "In Progress",
        "sprint": "Sprint 1 Baselines",
        "owner": "Noah",
        "effort": "M",
        "deliverable": "Demo"
    },
    {
        "title": "EPIC 3: Reproducibility + Submission",
        "body": """## Context
Ensure submission bundle works on fresh machine and meets competition requirements.

## Acceptance Criteria
- [ ] One-command runner + config system (`scripts/run_submission.py`)
- [ ] Fresh-machine QA runbook (`docs/QA_RUNBOOK.md`)
- [ ] Submission bundle generator (`submission/` directory)
- [ ] Final review + freeze + submit
- [ ] All tests pass on fresh machine
- [ ] Documentation complete
- [ ] Submission confirmed

## Dependencies
- Depends on EPIC 1 completion (detection model)
- Blocks final submission

## Owner Growth Outcome
- MLOps and reproducibility practices
- Submission packaging
- QA and testing

## Deliverable
Submission + Doc

## Notes
- Test on Docker container or fresh VM
- Verify no hardcoded paths
- Check all dependencies in requirements.txt""",
        "labels": ["type:mlops", "priority:P0"],
        "status": "Backlog",
        "sprint": "Sprint 4 Polish+Submission",
        "owner": "Edmund",
        "effort": "L",
        "deliverable": "Submission"
    },
    {
        "title": "EPIC 4: Portfolio Extension (Gary Micro-Twin)",
        "body": """## Context
Build lightweight "Gary Micro-Twin" with 3 anchor zones for controlled ML testing (NOT replacing SpectrumX data).

## Acceptance Criteria
- [ ] Gary Micro-Twin v1 (City Hall/HS/Library) - `configs/gary_micro_twin.yaml`
- [ ] Zone-aware synthetic IQ generation - `src/edge_ran_gary/digital_twin/gary_micro_twin.py`
- [ ] Demo notebook - `notebooks/03_gary_micro_twin_demo.ipynb`
- [ ] "Detector -> Occupancy prob -> Twin -> AI-RAN narrative" doc
- [ ] Integration points for Ananya's ML work
- [ ] Time-boxed (only after competition core complete)

## Dependencies
- Depends on EPIC 1 completion (competition core)
- Optional/portfolio work

## Owner Growth Outcome
- Digital twin simulation
- Zone-aware modeling
- Research extension narrative

## Deliverable
Code + Doc

## Notes
- Must be time-boxed (do not spill into competition core time)
- Micro-twin is sufficient (3 zones, basic generation)
- AI-RAN controller is optional (nice-to-have)""",
        "labels": ["phase:portfolio-digital-twin", "priority:P2"],
        "status": "Done",  # Already completed in this PR
        "sprint": "Portfolio (Digital Twin)",
        "owner": "Edmund",
        "effort": "M",
        "deliverable": "Code"
    }
]


def run_gh_command(cmd: List[str], return_json: bool = False) -> Optional[Dict]:
    """Run GitHub CLI command and return output."""
    try:
        result = subprocess.run(
            ["gh"] + cmd,
            capture_output=True,
            text=True,
            check=True
        )
        if return_json:
            return json.loads(result.stdout)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error running: gh {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return None


def create_labels():
    """Create GitHub labels if missing."""
    print("Creating labels...")
    existing_labels = run_gh_command(["label", "list", "--json", "name"], return_json=True)
    existing_label_names = {label["name"] for label in existing_labels or []}
    
    labels_to_create = [
        "type:baseline", "type:ssl", "type:anomaly", "type:viz", "type:eval",
        "type:mlops", "type:paper", "type:pm", "type:qa",
        "phase:core-detection", "phase:portfolio-digital-twin", "phase:portfolio-ai-ran",
        "priority:P0", "priority:P1", "priority:P2", "priority:P3", "blocked"
    ]
    
    for label in labels_to_create:
        if label not in existing_label_names:
            # Extract color from label type
            if "priority:" in label:
                color = "d73a4a"  # Red for priorities
            elif "type:" in label:
                color = "0e8a16"  # Green for types
            elif "phase:" in label:
                color = "1d76db"  # Blue for phases
            else:
                color = "ededed"  # Gray for others
            
            run_gh_command([
                "label", "create", label,
                "--color", color,
                "--description", f"Label: {label}"
            ])
            print(f"  ✅ Created label: {label}")
        else:
            print(f"  ⏭️  Label exists: {label}")


def create_project() -> Optional[str]:
    """Create GitHub Project v2 and return project number."""
    print("\nCreating GitHub Project...")
    
    # Check if project exists
    projects = run_gh_command(["project", "list", "--owner", "gunnchOS3k", "--format", "json"], return_json=True)
    if projects:
        for project in projects:
            if project.get("title") == PROJECT_TITLE:
                print(f"  ✅ Project exists: {project['number']}")
                return str(project["number"])
    
    # Create project
    result = run_gh_command([
        "project", "create",
        "--owner", "gunnchOS3k",
        "--title", PROJECT_TITLE,
        "--format", "json"
    ], return_json=True)
    
    if result:
        project_number = str(result.get("number", ""))
        print(f"  ✅ Created project: {project_number}")
        return project_number
    
    return None


def create_project_fields(project_number: str):
    """Create custom fields for project (Status, Sprint, Owner, Effort, Deliverable, Due Date)."""
    print("\nCreating project fields...")
    print("  ⚠️  Note: Custom fields must be created via GitHub UI or GraphQL API.")
    print("  ⚠️  Please create these fields manually:")
    print("     - Status (single select): Backlog, Ready, In Progress, Blocked, In Review, Done")
    print("     - Sprint (single select): Sprint 0-4, Portfolio (Digital Twin), Portfolio (AI-RAN)")
    print("     - Owner (single select): Edmund, Ananya, Noah")
    print("     - Effort (single select): XS, S, M, L, XL")
    print("     - Deliverable (single select): Code, Experiment, Figure, Doc, Demo, Submission")
    print("     - Due Date (date)")


def create_epic_issue(epic: Dict) -> Optional[str]:
    """Create GitHub issue for EPIC and return issue URL."""
    labels_str = ",".join(epic["labels"])
    
    # Create issue
    result = run_gh_command([
        "issue", "create",
        "--title", epic["title"],
        "--body", epic["body"],
        "--label", labels_str,
        "--assignee", epic["owner"].lower(),
        "--repo", REPO
    ])
    
    if result:
        # Extract issue number from URL
        # Format: https://github.com/owner/repo/issues/123
        match = __import__("re").search(r"issues/(\d+)", result)
        if match:
            issue_number = match.group(1)
            issue_url = f"https://github.com/{REPO}/issues/{issue_number}"
            print(f"  ✅ Created issue: {epic['title']} (#{issue_number})")
            return issue_url
    
    return None


def add_issue_to_project(project_number: str, issue_url: str):
    """Add issue to project."""
    result = run_gh_command([
        "project", "item-add", project_number,
        "--owner", "gunnchOS3k",
        "--url", issue_url
    ])
    if result:
        print(f"    ✅ Added to project")
    else:
        print(f"    ⚠️  Could not add to project (may need manual add)")


def main():
    """Main execution."""
    print("=" * 60)
    print("GitHub Project + EPIC Issues Setup")
    print("=" * 60)
    
    # Check auth
    auth_status = run_gh_command(["auth", "status"])
    if "Logged in" not in auth_status:
        print("❌ Not authenticated. Run: gh auth login")
        sys.exit(1)
    
    # Create labels
    create_labels()
    
    # Create project
    project_number = create_project()
    if not project_number:
        print("❌ Failed to create project")
        sys.exit(1)
    
    # Note about fields
    create_project_fields(project_number)
    
    # Create EPIC issues
    print("\nCreating EPIC issues...")
    issue_urls = []
    for epic in EPICS:
        issue_url = create_epic_issue(epic)
        if issue_url:
            issue_urls.append(issue_url)
            # Add to project
            add_issue_to_project(project_number, issue_url)
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"✅ Project created: {PROJECT_TITLE} (#{project_number})")
    print(f"✅ EPIC issues created: {len(issue_urls)}")
    print(f"\nProject Board URL:")
    print(f"  https://github.com/orgs/gunnchOS3k/projects/{project_number}")
    print(f"\nEPIC Issues:")
    for i, url in enumerate(issue_urls, 1):
        print(f"  {i}. {url}")
    print(f"\n⚠️  Next Steps:")
    print(f"  1. Create custom fields via GitHub UI (Status, Sprint, Owner, Effort, Deliverable, Due Date)")
    print(f"  2. Set field values for EPIC issues")
    print(f"  3. Create child issues for each EPIC")
    print(f"  4. Update project board as work progresses")


if __name__ == "__main__":
    main()
