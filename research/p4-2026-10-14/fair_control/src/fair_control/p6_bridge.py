"""Explicit application-outcome boundary over modeled completion candidates.

This does not change P5 dynamics or claim a closed-loop physical application
simulator. Independent final outcomes can reject a virtual work completion.
"""
from pathlib import Path
from dataclasses import replace,asdict
from .core import ContractError,TaskOutcome,completion_fraction
from .p6_application import run_reference,evaluate

def verify_modeled_completion(task,modeled_outcome,scratch:Path, *,network_up=False,released_s=0.):
    if modeled_outcome.task_id!=task.task_id:raise ContractError('Bridge task identity')
    generated=max(0.,modeled_outcome.finished_s-modeled_outcome.content_age_s)
    r=run_reference(task,scratch,virtual_now_s=modeled_outcome.finished_s,generated_s=generated,network_up=network_up)
    r=replace(r,quality=min(r.quality,modeled_outcome.quality))
    judged=evaluate(task,r,released_s=released_s)
    judged['model_candidate']=asdict(modeled_outcome);judged['raw_reference']=asdict(r)
    judged['interpretation']='Independent application verification over a virtual candidate; host execution is a constructed adapter check,not virtual/physical timing calibration.'
    return judged

def application_fraction(tasks,judgments,releases=None):
    ids=[x['outcome']['task_id'] for x in judgments]
    if len(ids)!=len(set(ids)) or not set(ids)<={t.task_id for t in tasks}:raise ContractError('Unknown/duplicate independent outcome')
    releases={} if releases is None else releases
    from .core import TaskRequirement
    requirements=[TaskRequirement(t.task_id,releases.get(t.task_id,0),releases.get(t.task_id,0)+t.deadline_s,t.freshness_limit_s,t.quality_minimum,t.requires_durable) for t in tasks]
    return completion_fraction(requirements,[TaskOutcome(**x['outcome']) for x in judgments])
