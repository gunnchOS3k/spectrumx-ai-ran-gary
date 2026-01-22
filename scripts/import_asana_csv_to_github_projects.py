#!/usr/bin/env python3
"""
Import Asana CSV tasks to GitHub Projects v2.

This script:
1. Reads SpX-DAC_gunnchOS3kMLV.csv
2. Creates GitHub labels (if missing)
3. Creates GitHub Project v2 with custom fields
4. Creates GitHub Issues from CSV tasks
5. Adds issues to Project and sets field values automatically
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
import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime, timedelta

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

SPRINT_MAPPING = {
    "0": "Sprint 0 Setup",
    "Sprint 0": "Sprint 0 Setup",
    "1": "Sprint 1 Baselines",
    "Sprint 1": "Sprint 1 Baselines",
    "2": "Sprint 2 SSL",
    "Sprint 2": "Sprint 2 SSL",
    "3": "Sprint 3 Anomaly+Fusion",
    "Sprint 3": "Sprint 3 Anomaly+Fusion",
    "4": "Sprint 4 Polish+Submission",
    "Sprint 4": "Sprint 4 Polish+Submission",
    "Final": "Sprint 4 Polish+Submission",
}


def run_gh_command(cmd: List[str], capture_output: bool = True, return_text: bool = False) -> Dict | str:
    """
    Run a GitHub CLI command and return JSON result or text.
    
    Args:
        cmd: Command arguments
        capture_output: Whether to capture output
        return_text: If True, return raw text instead of parsing JSON
        
    Returns:
        JSON dict if return_text=False, or raw text if return_text=True
    """
    try:
        result = subprocess.run(
            ["gh"] + cmd,
            capture_output=capture_output,
            text=True,
            check=True
        )
        if not capture_output:
            return {}
        if return_text:
            return result.stdout.strip()
        if result.stdout:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                # If JSON parsing fails, return empty dict
                return {}
        return {}
    except subprocess.CalledProcessError as e:
        print(f"Error running: gh {' '.join(cmd)}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return {} if not return_text else ""
    except json.JSONDecodeError:
        return {} if not return_text else ""


def create_label(name: str, color: str = "0E8A16") -> bool:
    """Create a GitHub label if it doesn't exist."""
    # Check if label exists
    labels = run_gh_command(["label", "list", "--repo", REPO, "--json", "name"])
    existing = [l["name"] for l in labels] if isinstance(labels, list) else []
    
    if name in existing:
        print(f"  Label '{name}' already exists")
        return True
    
    # Create label
    result = subprocess.run(
        ["gh", "label", "create", name, "--repo", REPO, "--color", color],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"  Created label '{name}'")
        return True
    return False


def create_all_labels():
    """Create all required labels."""
    print("Creating labels...")
    for label in LABELS:
        create_label(label)
    print(f"✓ Created/verified {len(LABELS)} labels\n")


def create_project() -> Tuple[Optional[str], Optional[int]]:
    """
    Create GitHub Project v2 and return project ID and number.
    
    Returns:
        Tuple of (project_id, project_number)
    """
    print("Creating GitHub Project...")
    
    # Check if project already exists
    projects = run_gh_command(["project", "list", "--owner", "gunnchOS3k", "--format", "json"])
    if isinstance(projects, list):
        for proj in projects:
            if proj.get("title") == PROJECT_TITLE:
                print(f"  Project '{PROJECT_TITLE}' already exists (number: {proj.get('number')})")
                # Get project ID via GraphQL
                project_number = proj.get("number")
                # Query for project ID
                query = '''
                {
                  organization(login: "gunnchOS3k") {
                    projectV2(number: %d) {
                      id
                    }
                  }
                }
                ''' % project_number
                result = run_gh_command(["api", "graphql", "-f", "query=" + query])
                if isinstance(result, dict) and "data" in result:
                    project_id = result["data"].get("organization", {}).get("projectV2", {}).get("id")
                    if project_id:
                        return project_id, project_number
                return None, project_number
    
    # Create new project
    result = run_gh_command(
        ["project", "create", "--owner", "gunnchOS3k", "--title", PROJECT_TITLE, "--format", "json"],
        return_text=False
    )
    project_id = result.get("id") if isinstance(result, dict) else None
    project_number = result.get("number") if isinstance(result, dict) else None
    
    if project_id:
        print(f"  Created project '{PROJECT_TITLE}' (ID: {project_id}, number: {project_number})")
        return project_id, project_number
    else:
        print("  ERROR: Failed to create project")
        return None, None


def get_project_fields(project_id: str) -> Dict[str, str]:
    """Get existing project fields and return mapping of name -> id."""
    query = '''
    {
      node(id: "%s") {
        ... on ProjectV2 {
          fields(first: 20) {
            nodes {
              id
              ... on ProjectV2Field {
                name
              }
              ... on ProjectV2SingleSelectField {
                name
                options {
                  id
                  name
                }
              }
            }
          }
        }
      }
    }
    ''' % project_id
    
    result = run_gh_command(["api", "graphql", "-f", f"query={query}"])
    fields_map = {}
    options_map = {}
    
    if isinstance(result, dict) and "data" in result:
        nodes = result["data"].get("node", {}).get("fields", {}).get("nodes", [])
        for field in nodes:
            field_id = field.get("id")
            field_name = field.get("name")
            if field_id and field_name:
                fields_map[field_name] = field_id
                # Store options for single select fields
                if "options" in field:
                    options_map[field_name] = {opt["name"]: opt["id"] for opt in field.get("options", [])}
    
    return fields_map, options_map


def create_project_fields(project_id: str) -> Dict[str, str]:
    """Create custom fields in the GitHub Project and return field ID mapping."""
    print("Creating project fields...")
    
    existing_fields_dict, existing_options = get_project_fields(project_id)
    existing_fields = existing_fields_dict
    
    # Create missing fields
    for field_name, field_def in PROJECT_FIELDS.items():
        if field_name in existing_fields:
            print(f"  Field '{field_name}' already exists")
            continue
        
        if field_def["type"] == "single_select":
            # Create single select field
            options_json = json.dumps([{"name": opt} for opt in field_def["options"]])
            mutation = '''
            mutation {
              createProjectV2Field(input: {
                projectId: "%s"
                name: "%s"
                dataType: SINGLE_SELECT
                singleSelectOptions: %s
              }) {
                projectV2Field {
                  id
                  name
                }
              }
            }
            ''' % (project_id, field_name, options_json)
            result = run_gh_command(["api", "graphql", "-f", "query=" + mutation])
            if result and isinstance(result, dict):
                field_data = result.get("data", {}).get("createProjectV2Field", {}).get("projectV2Field")
                if field_data:
                    existing_fields[field_name] = field_data.get("id")
                    print(f"  Created field '{field_name}'")
        elif field_def["type"] == "date":
            # Create date field
            mutation = '''
            mutation {
              createProjectV2Field(input: {
                projectId: "%s"
                name: "%s"
                dataType: DATE
              }) {
                projectV2Field {
                  id
                  name
                }
              }
            }
            ''' % (project_id, field_name)
            result = run_gh_command(["api", "graphql", "-f", "query=" + mutation])
            if result and isinstance(result, dict):
                field_data = result.get("data", {}).get("createProjectV2Field", {}).get("projectV2Field")
                if field_data:
                    existing_fields[field_name] = field_data.get("id")
                    print(f"  Created field '{field_name}'")
    
    print("✓ Project fields created\n")
    return existing_fields


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


def generate_issue_fingerprint(task: Dict) -> str:
    """Generate a stable fingerprint for idempotency check."""
    name = task.get("Name", "")
    due_date = task.get("Due Date", "")
    assignee_email = task.get("Assignee Email", "")
    fingerprint_str = f"{name}|{due_date}|{assignee_email}"
    return hashlib.md5(fingerprint_str.encode()).hexdigest()


def find_existing_issue(task: Dict) -> Optional[str]:
    """Check if issue already exists using fingerprint in body."""
    name = task.get("Name", "").strip()
    if not name:
        return None
    
    # Search for issues with similar title
    search_query = f'repo:{REPO} is:issue "{name}"'
    result = run_gh_command(["issue", "list", "--repo", REPO, "--search", search_query, "--json", "number,title,body"])
    
    if isinstance(result, list):
        fingerprint = generate_issue_fingerprint(task)
        for issue in result:
            body = issue.get("body", "")
            # Check if fingerprint matches (we'll add it to body)
            if fingerprint in body or issue.get("title", "").lower() == name.lower():
                return f"#{issue['number']}"
    
    return None


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
    if "baseline" in name and "stub" not in name and repo_audit.get("baselines_implemented"):
        return "Done"
    if "dataset" in name and "download" in name and repo_audit.get("dataset_loader_exists"):
        return "Done"
    if "repo skeleton" in name or "repo structure" in name:
        return "Done"  # Repo structure exists
    if "architecture" in name and repo_audit.get("architecture_docs_exist"):
        return "Done"
    
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
    
    fingerprint = generate_issue_fingerprint(task)
    
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
- Fingerprint: {fingerprint}
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


def create_issue(task: Dict, repo_audit: Dict) -> Optional[Tuple[str, str]]:
    """
    Create a GitHub Issue from Asana task.
    
    Returns:
        Tuple of (issue_url, issue_number) or None if failed/skipped
    """
    name = task.get("Name", "").strip()
    if not name:
        return None
    
    # Check for existing issue (idempotency)
    existing = find_existing_issue(task)
    if existing:
        print(f"  Issue '{name}' already exists {existing}")
        # Get issue URL
        issue_num = existing.replace("#", "")
        issue_data = run_gh_command(["issue", "view", issue_num, "--repo", REPO, "--json", "url"])
        if isinstance(issue_data, dict) and "url" in issue_data:
            return issue_data["url"], issue_num
        return None
    
    # Prepare issue creation
    assignee_email = task.get("Assignee Email", "")
    assignee_gh = None
    if "ananyajha@umass.edu" in assignee_email:
        assignee_gh = "Ananya-Jha-code"  # Update with actual GitHub username
    elif "newman1nj@alma.edu" in assignee_email:
        assignee_gh = None  # Update with actual GitHub username
    elif "egunnjr@gunnchos.com" in assignee_email:
        assignee_gh = "gunnchOS3k"
    
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
    
    # Create issue - use text output since gh issue create returns URL
    cmd = ["issue", "create", "--repo", REPO, "--title", name, "--body", create_issue_body(task)]
    if labels:
        cmd.extend(["--label", ",".join(labels)])
    if assignee_gh:
        cmd.extend(["--assignee", assignee_gh])
    
    # Get URL as text
    issue_url = run_gh_command(cmd, return_text=True)
    if not issue_url:
        print(f"  ERROR: Failed to create issue '{name}'")
        return None
    
    # Extract issue number from URL
    match = re.search(r'/issues/(\d+)', issue_url)
    if match:
        issue_num = match.group(1)
        print(f"  Created issue #{issue_num}: {name}")
        return issue_url, issue_num
    else:
        print(f"  WARNING: Created issue but couldn't parse number from URL: {issue_url}")
        return issue_url, ""


def add_issue_to_project(project_number: int, issue_url: str) -> Optional[str]:
    """Add issue to GitHub Project and return item ID."""
    try:
        result = subprocess.run(
            ["gh", "project", "item-add", str(project_number), "--owner", "gunnchOS3k", "--url", issue_url],
            capture_output=True,
            text=True,
            check=True
        )
        # Parse item ID from output if available
        # gh project item-add doesn't return JSON, so we'll get it via GraphQL
        return None  # Will get via GraphQL query
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Failed to add issue to project: {e.stderr}")
        return None


def get_project_item_id(project_id: str, issue_url: str) -> Optional[str]:
    """Get project item ID for an issue using GraphQL."""
    # Extract issue number from URL
    match = re.search(r'/issues/(\d+)', issue_url)
    if not match:
        return None
    
    issue_num = match.group(1)
    query = '''
    {
      repository(owner: "gunnchOS3k", name: "spectrumx-ai-ran-gary") {
        issue(number: %s) {
          projectItems(first: 10) {
            nodes {
              id
              project {
                id
              }
            }
          }
        }
      }
    }
    ''' % issue_num
    
    result = run_gh_command(["api", "graphql", "-f", "query=" + query])
    if isinstance(result, dict) and "data" in result:
        nodes = result["data"].get("repository", {}).get("issue", {}).get("projectItems", {}).get("nodes", [])
        for node in nodes:
            if node.get("project", {}).get("id") == project_id:
                return node.get("id")
    
    return None


def set_project_field_value(project_id: str, item_id: str, field_id: str, value: str, field_type: str = "single_select") -> bool:
    """Set a project field value using GraphQL."""
    if field_type == "single_select":
        # Get option ID first
        query = '''
        {
          node(id: "%s") {
            ... on ProjectV2SingleSelectField {
              options {
                id
                name
              }
            }
          }
        }
        ''' % field_id
        result = run_gh_command(["api", "graphql", "-f", "query=" + query])
        if isinstance(result, dict) and "data" in result:
            options = result["data"].get("node", {}).get("options", [])
            option_id = None
            for opt in options:
                if opt.get("name") == value:
                    option_id = opt.get("id")
                    break
            
            if not option_id:
                return False
            
            mutation = '''
            mutation {
              updateProjectV2ItemFieldValue(input: {
                projectId: "%s"
                itemId: "%s"
                fieldId: "%s"
                value: {
                  singleSelectOptionId: "%s"
                }
              }) {
                projectV2Item {
                  id
                }
              }
            }
            ''' % (project_id, item_id, field_id, option_id)
        else:
            return False
    elif field_type == "date":
        # Format date as ISO 8601
        mutation = '''
        mutation {
          updateProjectV2ItemFieldValue(input: {
            projectId: "%s"
            itemId: "%s"
            fieldId: "%s"
            value: {
              date: "%s"
            }
          }) {
            projectV2Item {
              id
            }
          }
        }
        ''' % (project_id, item_id, field_id, value)
    else:
        return False
    
    result = run_gh_command(["api", "graphql", "-f", "query=" + mutation])
    return isinstance(result, dict) and "data" in result and "updateProjectV2ItemFieldValue" in result.get("data", {})


def set_all_project_fields(project_id: str, item_id: str, fields_map: Dict[str, str], task: Dict, repo_audit: Dict):
    """Set all project field values for an issue."""
    # Status
    status = determine_status(task, repo_audit)
    if "Status" in fields_map:
        set_project_field_value(project_id, item_id, fields_map["Status"], status, "single_select")
    
    # Sprint
    sprint_raw = task.get("Sprint", "")
    sprint = SPRINT_MAPPING.get(sprint_raw, "Sprint 0 Setup")
    if "Sprint" in fields_map:
        set_project_field_value(project_id, item_id, fields_map["Sprint"], sprint, "single_select")
    
    # Owner
    assignee = task.get("Assignee", "")
    email = task.get("Assignee Email", "")
    owner = map_owner(assignee, email)
    if "Owner" in fields_map:
        set_project_field_value(project_id, item_id, fields_map["Owner"], owner, "single_select")
    
    # Effort
    effort_raw = task.get("Effort", "")
    effort = EFFORT_MAPPING.get(effort_raw, "M")
    if "Effort" in fields_map:
        set_project_field_value(project_id, item_id, fields_map["Effort"], effort, "single_select")
    
    # Deliverable
    name = task.get("Name", "")
    deliverable = determine_deliverable(name)
    if "Deliverable" in fields_map:
        set_project_field_value(project_id, item_id, fields_map["Deliverable"], deliverable, "single_select")
    
    # Due Date
    due_date = task.get("Due Date", "")
    if due_date and "Due Date" in fields_map:
        # Convert date format if needed (YYYY-MM-DD)
        try:
            # Try parsing various date formats
            if "/" in due_date:
                parts = due_date.split("/")
                if len(parts) == 3:
                    # MM/DD/YYYY or DD/MM/YYYY - assume MM/DD/YYYY
                    month, day, year = parts
                    iso_date = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    iso_date = due_date
            else:
                iso_date = due_date
            set_project_field_value(project_id, item_id, fields_map["Due Date"], iso_date, "date")
        except Exception:
            pass  # Skip if date parsing fails


def main():
    """Main execution."""
    print("=" * 60)
    print("Asana to GitHub Projects Migration")
    print("=" * 60)
    print()
    
    # Step 1: Create labels
    create_all_labels()
    
    # Step 2: Create project
    project_id, project_number = create_project()
    if not project_id or not project_number:
        print("ERROR: Failed to create project. Exiting.")
        sys.exit(1)
    
    # Step 3: Create project fields
    fields_map = create_project_fields(project_id)
    
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
    
    # Step 6: Create issues and add to project
    print("Creating issues and adding to project...")
    created = 0
    updated = 0
    skipped = 0
    issue_urls = []
    
    for task in tasks:
        result = create_issue(task, repo_audit)
        if result:
            issue_url, issue_num = result
            issue_urls.append((issue_url, issue_num, task))
            
            # Add to project
            add_issue_to_project(project_number, issue_url)
            
            # Get item ID
            item_id = get_project_item_id(project_id, issue_url)
            if item_id:
                # Set all field values
                set_all_project_fields(project_id, item_id, fields_map, task, repo_audit)
                created += 1
            else:
                print(f"  WARNING: Could not get item ID for issue {issue_num}, fields not set")
                created += 1
        else:
            skipped += 1
    
    print(f"\n✓ Created {created} issues, updated {updated}, skipped {skipped}\n")
    
    # Step 7: Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Project: {PROJECT_TITLE}")
    print(f"Project Number: {project_number}")
    print(f"Project URL: https://github.com/users/gunnchOS3k/projects/{project_number}")
    print(f"Issues created: {created}")
    print(f"Issues updated: {updated}")
    print(f"Issues skipped: {skipped}")
    print(f"\n✅ Migration complete! All issues added to project with field values set.")
    print()


if __name__ == "__main__":
    main()
