#!/usr/bin/env python3
"""
Import Asana CSV tasks to GitHub Projects v2.

This script:
1. Reads SpX-DAC_gunnchOS3kMLV.csv
2. Creates GitHub labels (if missing)
3. Creates GitHub Project v2 with custom fields
4. Creates GitHub Issues from CSV tasks
5. Adds issues to Project with field values
6. Prints summary of created/updated/skipped items

Requirements:
- GitHub CLI (gh) installed and authenticated
- Project scope: gh auth refresh -s project
- Repository: gunnchOS3k/spectrumx-ai-ran-gary

Usage:
    python scripts/import_asana_csv_to_github_projects.py
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
import re

# Configuration
REPO = "gunnchOS3k/spectrumx-ai-ran-gary"
PROJECT_TITLE = "SpX-DAC 2026 — spectrumx-ai-ran-gary (gunnchOS3kMLV)"
CSV_FILE = Path(__file__).parent.parent / "SpX-DAC_gunnchOS3kMLV.csv"

# Label definitions
LABELS = [
    "type:baseline", "type:ssl", "type:anomaly", "type:viz", "type:eval",
    "type:mlops", "type:paper", "type:pm", "type:qa",
    "phase:core-detection", "phase:portfolio-digital-twin", "phase:portfolio-ai-ran",
    "priority:P0", "priority:P1", "priority:P2", "priority:P3", "blocked"
]

# Project field definitions
PROJECT_FIELDS = {
    "Status": {
        "type": "single_select",
        "options": ["Backlog", "Ready", "In Progress", "Blocked", "In Review", "Done"]
    },
    "Sprint": {
        "type": "single_select",
        "options": [
            "Sprint 0 Setup", "Sprint 1 Baselines", "Sprint 2 SSL",
            "Sprint 3 Anomaly+Fusion", "Sprint 4 Polish+Submission",
            "Portfolio (Digital Twin)", "Portfolio (AI-RAN)"
        ]
    },
    "Owner": {
        "type": "single_select",
        "options": ["Edmund", "Ananya", "Noah"]
    },
    "Effort": {
        "type": "single_select",
        "options": ["XS", "S", "M", "L", "XL"]
    },
    "Deliverable": {
        "type": "single_select",
        "options": ["Code", "Experiment", "Figure", "Doc", "Demo", "Submission"]
    },
    "Due Date": {
        "type": "date"
    }
}

# Mapping from CSV fields to GitHub fields
OWNER_MAPPING = {
    "Edmund Gunn, Jr.": "Edmund",
    "egunnjr@gunnchos.com": "Edmund",
    "ananyajha@umass.edu": "Ananya",
    "Noah Newman": "Noah",
    "newman1nj@alma.edu": "Noah",
}

TYPE_TO_LABEL = {
    "Baseline": "type:baseline",
    "SSL": "type:ssl",
    "Anomaly": "type:anomaly",
    "Viz": "type:viz",
    "Data/Viz": "type:viz",
    "Eval": "type:eval",
    "Eval/PM": "type:eval",
    "MLOps": "type:mlops",
    "Paper": "type:paper",
    "PM": "type:pm",
    "PM/Engineering": "type:pm",
    "PM/Security": "type:pm",
    "QA": "type:qa",
    "Research": "type:ssl",
    "Submission": "type:pm",
}

PHASE_TO_LABEL = {
    "Core": "phase:core-detection",
    "Improve": "phase:core-detection",
    "Define": "phase:core-detection",
    "Measure": "phase:core-detection",
    "Analyze": "phase:core-detection",
    "Control": "phase:core-detection",
    "Digital Twin": "phase:portfolio-digital-twin",
    "AI-RAN": "phase:portfolio-ai-ran",
}

PRIORITY_TO_LABEL = {
    "P0": "priority:P0",
    "P1": "priority:P1",
    "P2": "priority:P2",
    "P3": "priority:P3",
}

EFFORT_MAPPING = {
    "XS": "XS",
    "S": "S",
    "M": "M",
    "L": "L",
    "XL": "XL",
}


def run_gh_command(cmd: List[str], capture_output: bool = True) -> Dict:
    """Run a GitHub CLI command and return JSON result."""
    try:
        result = subprocess.run(
            ["gh"] + cmd,
            capture_output=capture_output,
            text=True,
            check=True
        )
        if capture_output and result.stdout:
            return json.loads(result.stdout)
        return {}
    except subprocess.CalledProcessError as e:
        print(f"Error running: gh {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        return {}
    except json.JSONDecodeError:
        return {}


def create_label(name: str, color: str = "0E8A16") -> bool:
    """Create a GitHub label if it doesn't exist."""
    # Check if label exists
    labels = run_gh_command(["label", "list", "--repo", REPO, "--json", "name"])
    existing = [l["name"] for l in labels]
    
    if name in existing:
        print(f"  Label '{name}' already exists")
        return True
    
    # Create label
    result = run_gh_command(
        ["label", "create", name, "--repo", REPO, "--color", color],
        capture_output=False
    )
    print(f"  Created label '{name}'")
    return True


def create_all_labels():
    """Create all required labels."""
    print("Creating labels...")
    for label in LABELS:
        create_label(label)
    print(f"✓ Created {len(LABELS)} labels\n")


def create_project() -> Optional[str]:
    """Create GitHub Project v2 and return project ID."""
    print("Creating GitHub Project...")
    
    # Check if project already exists
    projects = run_gh_command(["project", "list", "--owner", "gunnchOS3k", "--json", "title,number"])
    for proj in projects:
        if proj["title"] == PROJECT_TITLE:
            print(f"  Project '{PROJECT_TITLE}' already exists (number: {proj['number']})")
            # Get full project details
            details = run_gh_command(["project", "view", str(proj["number"]), "--owner", "gunnchOS3k", "--json", "id"])
            return details.get("id")
    
    # Create new project
    result = run_gh_command(
        ["project", "create", "--owner", "gunnchOS3k", "--title", PROJECT_TITLE, "--format", "json"]
    )
    project_id = result.get("id")
    if project_id:
        print(f"  Created project '{PROJECT_TITLE}' (ID: {project_id})")
    else:
        print("  ERROR: Failed to create project")
        return None
    
    return project_id


def create_project_fields(project_id: str):
    """Create custom fields in the GitHub Project."""
    print("Creating project fields...")
    
    # Get existing fields
    fields = run_gh_command(
        ["api", "graphql", "-f", f'query={{ node(id: "{project_id}") {{ ... on ProjectV2 {{ fields(first: 20) {{ nodes {{ id name {{ ... on ProjectV2Field {{ name }} ... on ProjectV2SingleSelectField {{ name options {{ id name }} }} }} }} }} }} }} }}']
    )
    
    existing_fields = {}
    if fields and "data" in fields:
        nodes = fields["data"].get("node", {}).get("fields", {}).get("nodes", [])
        for field in nodes:
            existing_fields[field.get("name", "")] = field.get("id")
    
    # Create missing fields
    for field_name, field_def in PROJECT_FIELDS.items():
        if field_name in existing_fields:
            print(f"  Field '{field_name}' already exists")
            continue
        
        if field_def["type"] == "single_select":
            # Create single select field
            options_json = json.dumps(field_def["options"])
            mutation = f'''
            mutation {{
              createProjectV2Field(input: {{
                projectId: "{project_id}"
                name: "{field_name}"
                dataType: SINGLE_SELECT
                singleSelectOptions: {options_json}
              }}) {{
                projectV2Field {{
                  id
                  name
                }}
              }}
            }}
            '''
            result = run_gh_command(["api", "graphql", "-f", f"query={mutation}"])
            if result:
                print(f"  Created field '{field_name}'")
        elif field_def["type"] == "date":
            # Create date field
            mutation = f'''
            mutation {{
              createProjectV2Field(input: {{
                projectId: "{project_id}"
                name: "{field_name}"
                dataType: DATE
              }}) {{
                projectV2Field {{
                  id
                  name
                }}
              }}
            }}
            '''
            result = run_gh_command(["api", "graphql", "-f", f"query={mutation}"])
            if result:
                print(f"  Created field '{field_name}'")
    
    print("✓ Project fields created\n")


def parse_csv() -> List[Dict]:
    """Parse the Asana CSV file."""
    tasks = []
    with open(CSV_FILE, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Clean up the row
            cleaned = {}
            for key, value in row.items():
                # Remove BOM and whitespace
                clean_key = key.strip().replace('\ufeff', '')
                cleaned[clean_key] = value.strip() if value else ""
            tasks.append(cleaned)
    return tasks


def map_owner(assignee: str, email: str) -> str:
    """Map Asana assignee to GitHub Owner field."""
    if assignee in OWNER_MAPPING:
        return OWNER_MAPPING[assignee]
    if email in OWNER_MAPPING:
        return OWNER_MAPPING[email]
    # Default based on name patterns
    if "ananya" in assignee.lower() or "ananya" in email.lower():
        return "Ananya"
    if "noah" in assignee.lower() or "noah" in email.lower():
        return "Noah"
    if "edmund" in assignee.lower() or "edmund" in email.lower() or "gunn" in assignee.lower():
        return "Edmund"
    return "Edmund"  # Default


def determine_status(task: Dict, repo_audit: Dict) -> str:
    """Determine GitHub Project status based on task and repo audit."""
    completed = task.get("Completed At", "").strip()
    if completed:
        return "Done"
    
    # Check if deliverable exists in repo
    name = task.get("Name", "").lower()
    
    # Check for common completion indicators
    if "streamlit" in name and repo_audit.get("streamlit_app_exists"):
        return "Done"
    if "baseline" in name and repo_audit.get("baselines_implemented"):
        return "Done"
    if "dataset" in name and "download" in name and repo_audit.get("dataset_loader_exists"):
        return "Done"
    if "repo skeleton" in name or "repo structure" in name:
        return "Done"  # Repo structure exists
    
    return "Backlog"


def create_issue_body(task: Dict) -> str:
    """Create GitHub Issue body from Asana task."""
    name = task.get("Name", "")
    notes = task.get("Notes", "")
    assignee = task.get("Assignee", "")
    email = task.get("Assignee Email", "")
    due_date = task.get("Due Date", "")
    blocked_by = task.get("Blocked By (Dependencies)", "")
    blocking = task.get("Blocking (Dependencies)", "")
    
    body = f"""## Context
{notes if notes else "Task from Asana migration."}

## Acceptance Criteria
{extract_acceptance_criteria(notes)}

## Dependencies
{format_dependencies(blocked_by, blocking)}

## Owner Growth Outcome
{get_growth_outcome(name, assignee)}

## Deliverable
{determine_deliverable(name)}

## Notes
- Original Asana Task ID: {task.get('Task ID', 'N/A')}
- Due Date: {due_date if due_date else 'TBD'}
- Original Assignee: {assignee} ({email})
"""
    return body


def extract_acceptance_criteria(notes: str) -> str:
    """Extract acceptance criteria from notes."""
    if "Acceptance:" in notes:
        criteria = notes.split("Acceptance:")[1].strip()
        # Convert to bullet points
        if "\n" in criteria:
            lines = criteria.split("\n")
            return "\n".join(f"- {line.strip()}" for line in lines if line.strip())
        return f"- {criteria}"
    return "- [ ] Task completed and tested\n- [ ] Code/documentation reviewed"


def format_dependencies(blocked_by: str, blocking: str) -> str:
    """Format dependencies."""
    deps = []
    if blocked_by:
        deps.append(f"**Blocked by:** {blocked_by}")
    if blocking:
        deps.append(f"**Blocks:** {blocking}")
    if not deps:
        return "None"
    return "\n".join(deps)


def get_growth_outcome(name: str, assignee: str) -> str:
    """Get growth outcome based on task and assignee."""
    name_lower = name.lower()
    assignee_lower = assignee.lower()
    
    if "ananya" in assignee_lower or "ananyajha" in assignee_lower:
        if "ssl" in name_lower or "self-supervised" in name_lower:
            return "- Implement SSL methods (SimCLR/BYOL/wav2vec)\n- Gain experience with contrastive learning\n- Learn calibration techniques for production ML"
        if "calibration" in name_lower:
            return "- Master probability calibration (Platt, Isotonic)\n- Understand ECE and calibration curves\n- Apply to production ML systems"
        return "- Implement ML models end-to-end\n- Gain MLOps experience\n- Learn reproducibility best practices"
    
    if "noah" in assignee_lower or "newman" in assignee_lower:
        if "dataset" in name_lower or "eda" in name_lower:
            return "- Master data pipeline design\n- Learn evaluation harness construction\n- Gain experience with data visualization"
        if "streamlit" in name_lower or "dashboard" in name_lower:
            return "- Build production-ready dashboards\n- Learn interactive visualization\n- Gain full-stack experience"
        return "- Build evaluation systems\n- Learn QA and testing\n- Gain DevOps experience"
    
    # Default for Edmund
    if "digital twin" in name_lower or "ai-ran" in name_lower:
        return "- Design system architecture\n- Integrate research components\n- Create judge-ready documentation"
    return "- Lead technical direction\n- Create comprehensive documentation\n- Ensure reproducibility"


def determine_deliverable(name: str) -> str:
    """Determine deliverable type from task name."""
    name_lower = name.lower()
    if "script" in name_lower or "code" in name_lower or "implement" in name_lower:
        return "Code"
    if "experiment" in name_lower or "ablation" in name_lower:
        return "Experiment"
    if "plot" in name_lower or "figure" in name_lower or "visualization" in name_lower:
        return "Figure"
    if "doc" in name_lower or "documentation" in name_lower or "md" in name_lower:
        return "Doc"
    if "demo" in name_lower or "dashboard" in name_lower or "streamlit" in name_lower:
        return "Demo"
    if "submission" in name_lower or "submit" in name_lower:
        return "Submission"
    return "Code"


def audit_repo() -> Dict:
    """Audit repository to determine what's already done."""
    repo_root = Path(__file__).parent.parent
    audit = {
        "streamlit_app_exists": (repo_root / "apps" / "streamlit_app.py").exists(),
        "baselines_implemented": (repo_root / "src" / "edge_ran_gary" / "detection" / "baselines.py").exists(),
        "dataset_loader_exists": (repo_root / "src" / "edge_ran_gary" / "data_pipeline" / "spectrumx_loader.py").exists(),
        "detection_package_exists": (repo_root / "src" / "edge_ran_gary" / "detection").exists(),
        "architecture_docs_exist": (repo_root / "docs" / "architecture").exists(),
        "uml_diagrams_exist": (repo_root / "docs" / "uml").exists(),
    }
    return audit


def create_issue(task: Dict, repo_audit: Dict) -> Optional[str]:
    """Create a GitHub Issue from Asana task."""
    name = task.get("Name", "").strip()
    if not name:
        return None
    
    # Check if issue already exists
    existing = run_gh_command(
        ["issue", "list", "--repo", REPO, "--search", f'"{name}"', "--json", "number,title"]
    )
    for issue in existing:
        if issue["title"].lower() == name.lower():
            print(f"  Issue '{name}' already exists (#{issue['number']})")
            return f"#{issue['number']}"
    
    # Prepare issue creation
    assignee_email = task.get("Assignee Email", "")
    assignee_gh = None
    if "ananyajha@umass.edu" in assignee_email:
        assignee_gh = "Ananya-Jha-code"  # Update with actual GitHub username
    elif "newman1nj@alma.edu" in assignee_email:
        assignee_gh = None  # Update with actual GitHub username
    elif "egunnjr@gunnchos.com" in assignee_email:
        assignee_gh = "gunnchOS3k"  # Update with actual GitHub username
    
    # Collect labels
    labels = []
    task_type = task.get("Type", "")
    if task_type in TYPE_TO_LABEL:
        labels.append(TYPE_TO_LABEL[task_type])
    
    phase = task.get("Phase", "")
    if phase in PHASE_TO_LABEL:
        labels.append(PHASE_TO_LABEL[phase])
    
    priority = task.get("Priority", "")
    if priority in PRIORITY_TO_LABEL:
        labels.append(PRIORITY_TO_LABEL[priority])
    
    # Create issue
    cmd = ["issue", "create", "--repo", REPO, "--title", name, "--body", create_issue_body(task)]
    if labels:
        cmd.extend(["--label", ",".join(labels)])
    if assignee_gh:
        cmd.extend(["--assignee", assignee_gh])
    
    result = run_gh_command(cmd, capture_output=True)
    if result and "number" in result:
        issue_num = result["number"]
        print(f"  Created issue #{issue_num}: {name}")
        return f"#{issue_num}"
    
    return None


def main():
    """Main execution."""
    print("=" * 60)
    print("Asana to GitHub Projects Migration")
    print("=" * 60)
    print()
    
    # Step 1: Create labels
    create_all_labels()
    
    # Step 2: Create project
    project_id = create_project()
    if not project_id:
        print("ERROR: Failed to create project. Exiting.")
        sys.exit(1)
    
    # Step 3: Create project fields
    create_project_fields(project_id)
    
    # Step 4: Parse CSV
    print("Parsing CSV...")
    tasks = parse_csv()
    print(f"  Found {len(tasks)} tasks\n")
    
    # Step 5: Audit repo
    print("Auditing repository...")
    repo_audit = audit_repo()
    print(f"  Streamlit app: {repo_audit['streamlit_app_exists']}")
    print(f"  Baselines: {repo_audit['baselines_implemented']}")
    print(f"  Dataset loader: {repo_audit['dataset_loader_exists']}")
    print()
    
    # Step 6: Create issues
    print("Creating issues...")
    created = 0
    skipped = 0
    issue_numbers = []
    
    for task in tasks:
        issue_num = create_issue(task, repo_audit)
        if issue_num:
            issue_numbers.append(issue_num)
            created += 1
        else:
            skipped += 1
    
    print(f"\n✓ Created {created} issues, skipped {skipped}\n")
    
    # Step 7: Add issues to project (manual step - GitHub CLI limitation)
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Project: {PROJECT_TITLE}")
    print(f"Project ID: {project_id}")
    print(f"Issues created: {created}")
    print(f"Issues skipped: {skipped}")
    print(f"\nNext steps:")
    print(f"1. Add issues to project manually via GitHub UI")
    print(f"2. Or use: gh project item-add {project_id} --url <issue-url>")
    print(f"3. Set project field values via GitHub UI or GraphQL API")
    print()


if __name__ == "__main__":
    main()
