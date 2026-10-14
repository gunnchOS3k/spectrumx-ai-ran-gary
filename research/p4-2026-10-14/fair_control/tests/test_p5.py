"""P5 constructed, analytical and deliberately invalid input checks. No inference."""
import unittest,copy,json,hashlib,tempfile
from dataclasses import replace,asdict
from pathlib import Path
from unittest.mock import patch
from fair_control.core import *
from fair_control.tasks import *
from fair_control.dynamics import *
from fair_control.controllers import *
from fair_control.calibration import *
from fair_control.runner import validate_fixture,run,validate_output,tree_manifest,verify_manifest,AUTH,ROOT

class P5(unittest.TestCase):
    def spec(self,id='a',work=4,deadline=10,**kw):
        req=TaskRequirement(id,kw.pop('released_s',0),deadline,kw.pop('max_age_s',100),kw.pop('minimum_quality',1),kw.pop('requires_durable',True))
        return TaskSpec(req,work,kw.pop('memory_mb',1),'abc',kw.pop('expected_sha256',sha('abc')),**kw)
    def truth(self,env,duration=1,up=True,cpu=10,memory=100,power=2,energy=100,capacity=10):
        return TruthFrame('frame',tuple(env.tasks),(PathTruth('TN',up,capacity),PathTruth('NTN',up,capacity)),(('local',cpu),('edge',cpu),('cloud',cpu)),Resources(capacity,cpu,memory,power,energy),duration)
    def action(self,id='a',place='local',cpu=4,bw=0,power=1,memory=1,fidelity='target',network='TN'):
        return Action(None if place=='local' else network,place,(Allocation(id,bw,cpu,memory,power),),fidelity)
    def step(self,e,a,**kw):return e.step(self.truth(e,**kw),a,{'TN':0,'NTN':0})
    def estimates(self,**kw):
        x={'timestamp_s':0.,'estimated_paths':[{'name':'TN','up':True,'capacity_mbps':10.,'latency_s':0.}], 'task_queue':[{'task_id':'a','remaining_mcycles':4.,'remaining_mbit':0.,'memory_mb':1.,'deadline_s':10.,'minimum_quality':1.,'content_age_s':0.,'max_age_s':100.,'local_eligible':True,'network_required':False}], 'remaining_energy_j':100.,'previous_action':None,'forecasts':[{'timestamp_s':1.,'duration_s':1.,'paths':[{'name':'TN','up':True,'capacity_mbps':10.,'latency_s':0.}]},{'timestamp_s':2.,'duration_s':1.,'paths':[{'name':'TN','up':True,'capacity_mbps':10.,'latency_s':0.}]}]}
        x.update(kw);return x
    def test_offline_success(self):
        e=Environment([self.spec()],10);self.step(e,self.action(),up=False);self.assertEqual(e.evaluator.fraction(e.observed),1)
    def test_connected_wrong_bytes_fail(self):
        e=Environment([self.spec(expected_sha256=sha('different'))],10);self.step(e,self.action(place='edge'));self.assertEqual(e.tasks['a'].failure,'outcome_rejected');self.assertEqual(e.observed,[])
    def test_unsupported_workload(self):
        e=Environment([self.spec(operation='server_only')],10);self.step(e,self.action());self.assertEqual(e.tasks['a'].failure,'unsupported_workload')
    def test_deadline_miss(self):
        e=Environment([self.spec(work=5,deadline=1)],10);self.step(e,self.action(cpu=4));self.assertEqual(e.tasks['a'].failure,'deadline')
    def test_energy_exhaustion(self):
        e=Environment([self.spec()],.5);r=self.step(e,self.action());self.assertIn('energy_exceeded',r['errors']);self.assertEqual(e.energy_j,.5);self.assertEqual(e.tasks['a'].progress_mcycles,0)
    def test_memory_shortfall(self):
        e=Environment([self.spec(memory_mb=2)],10);r=self.step(e,self.action(memory=1));self.assertIn('task_memory_requirement',r['errors'])
    def test_checkpoint_progress_preserved(self):
        e=Environment([self.spec(work=10)],10);self.step(e,self.action(cpu=2));c=e.checkpoint('a',.1,.01);e.crash('a');self.assertEqual(e.tasks['a'].progress_mcycles,0);e.restore('a',.2,.02);self.assertEqual(e.tasks['a'].progress_mcycles,c['progress_mcycles']);self.assertAlmostEqual(e.now_s,1.3)
    def test_restore_without_checkpoint_rejected(self):
        e=Environment([self.spec(work=10)],10);e.crash('a');self.assertRaises(ContractError,e.restore,'a',.1,0)
    def test_restore_zero_time_rejected(self):
        e=Environment([self.spec(work=10)],10);e.checkpoint('a',.1,0);e.crash('a');self.assertRaises(ContractError,e.restore,'a',0,0)
    def test_corrupt_checkpoint_rejected(self):
        e=Environment([self.spec(work=10)],10);e.checkpoint('a',.1,0);e.crash('a');e.tasks['a'].checkpoint['progress_mcycles']=10;self.assertRaises(ContractError,e.restore,'a',.1,0)
    def test_cross_task_checkpoint_rejected(self):
        e=Environment([self.spec(),self.spec('b')],10);c=e.checkpoint('a',.1,0);e.crash('b');e.tasks['b'].checkpoint=c;self.assertRaises(ContractError,e.restore,'b',.1,0)
    def test_restore_deadline_failure(self):
        e=Environment([self.spec(work=10,deadline=1)],10);e.checkpoint('a',.1,0);e.crash('a');self.assertRaises(ContractError,e.restore,'a',1,0);self.assertEqual(e.tasks['a'].failure,'deadline')
    def test_stale_content(self):
        e=Environment([self.spec(content_keys=('x',),max_age_s=.1)],10,[Content('x','bytes',0)]);self.step(e,self.action());self.assertEqual(e.tasks['a'].failure,'outcome_rejected')
    def test_missing_content_block(self):
        e=Environment([self.spec(content_keys=('missing',))],10);self.step(e,self.action());self.assertEqual(e.tasks['a'].state,State.BLOCKED)
    def test_connectivity_loss_local_progress_completes(self):
        e=Environment([self.spec(work=8)],10);self.step(e,self.action(),up=True);self.step(e,self.action(),up=False);self.assertEqual(e.tasks['a'].state,State.COMPLETED)
    def test_terminal_no_more_execution(self):
        e=Environment([self.spec()],10);self.step(e,self.action());before=e.energy_j;r=self.step(e,self.action());self.assertIn('inactive_task',r['errors']);self.assertEqual(e.energy_j,before);self.assertEqual(len(e.observed),1)
    def test_invalid_state_transition(self):
        t=TaskRuntime();self.assertRaises(ContractError,t.move,State.COMPLETED,0,'shortcut')
    def test_arrival_queue_and_boundary(self):
        e=Environment([self.spec('a',released_s=1)],10);self.assertEqual(e.queue_length,0);self.step(e,Action.defer());self.assertEqual(e.tasks['a'].state,State.READY);self.assertEqual(e.queue_length,1)
    def test_midframe_arrival_requires_split(self):
        e=Environment([self.spec(released_s=.5)],10);self.assertRaises(ContractError,self.step,e,Action.defer())
    def test_resources_enforced(self):
        for kw,code in [({'cpu':11},'over_cpu_mcycles_s'),({'power':3},'over_power_w'),({'memory':101},'over_memory_mb'),({'place':'edge','bw':11},'over_bandwidth_mbps')]:
            with self.subTest(kw=kw):
                e=Environment([self.spec()],100);r=self.step(e,self.action(**kw));self.assertIn(code,r['errors']);self.assertEqual(e.energy_j,100)
    def test_network_failure_rejected(self):
        e=Environment([self.spec()],10);r=self.step(e,self.action(place='edge'),up=False);self.assertIn('unavailable_path',r['errors']);self.assertEqual(e.tasks['a'].progress_mcycles,0)
    def test_local_network_requirement_rejected(self):
        e=Environment([self.spec(network_required=True)],10);r=self.step(e,self.action());self.assertIn('local_ineligible',r['errors'])
    def test_latency_delivery_math(self):
        e=Environment([self.spec(work=2,input_mbit=2,local_eligible=False)],10)
        r=e.step(self.truth(e,duration=2),self.action(place='edge',cpu=2,bw=2),{'TN':.5,'NTN':0});self.assertEqual(e.tasks['a'].progress_mcycles,1);self.assertEqual(r['delivered_mbit'],2);self.assertEqual(r['cpu_mcycles'],1)
    def test_energy_w_times_s(self):
        e=Environment([self.spec(work=100)],10);r=self.step(e,self.action(power=2),duration=2);self.assertEqual(r['energy_j'],4);self.assertEqual(e.energy_j,6);self.assertEqual(power_duration_energy(2,2),4)
    def test_switch_stall_changes_service(self):
        e=Environment([self.spec(work=100)],10,model=Model(network_stall_s=.5,network_energy_j=.25));self.step(e,self.action(place='edge',network='TN',cpu=2));r=self.step(e,self.action(place='edge',network='NTN',cpu=2));self.assertEqual(r['stall_s'],.5);self.assertEqual(r['cpu_mcycles'],1);self.assertEqual(r['energy_j'],1.25)
    def test_migration_resets_volatile_progress(self):
        e=Environment([self.spec(work=100)],10,model=Model(placement_stall_s=.5,network_stall_s=0));self.step(e,self.action(cpu=2));self.step(e,self.action(place='edge',cpu=2));self.assertEqual(e.tasks['a'].progress_mcycles,1)
    def test_long_switch_rejected_consistently(self):
        e=Environment([self.spec(work=100)],10,model=Model(network_stall_s=1));self.step(e,self.action(place='edge'));r=self.step(e,self.action(place='edge',network='NTN'));self.assertIn('transition_consumes_interval',r['errors'])
    def test_queue_and_no_energy_creation_restart_mutations(self):
        e=Environment([self.spec()],10);s=e.snapshot();s['energy_j']=11;self.assertRaises(ContractError,Environment.restart,e.specs,s)
        s=e.snapshot();s['tasks']['a']['progress_mcycles']=-1;self.assertRaises(ContractError,Environment.restart,e.specs,s)
    def test_no_duplicate_completion(self):
        e=Environment([self.spec()],10);self.step(e,self.action());s=e.snapshot();s['observed']*=2;self.assertRaises(ContractError,Environment.restart,e.specs,s)
    def test_deterministic_restart_and_replay(self):
        e=Environment([self.spec(work=8)],10);self.step(e,self.action());e2=Environment.restart(e.specs,json.loads(canonical(e.snapshot())));self.step(e,self.action());self.step(e2,self.action());self.assertEqual(canonical(e.snapshot()),canonical(e2.snapshot()))
    def test_restart_contract_wrong(self):
        e=Environment([self.spec()],10);self.assertRaises(ContractError,Environment.restart,[self.spec(work=99)],e.snapshot())
    def test_independent_oracle_ignores_score(self):
        e=Evaluator([self.spec()]);o=Observed('a',1,'wrong',0,1,True);self.assertEqual(e.fraction([o]),0);self.assertEqual(e.fraction([replace(o,output='abc')]),1)
    def test_success_requires_quality_freshness_durable(self):
        e=Evaluator([self.spec(max_age_s=1)]);o=Observed('a',1,'abc',0,1,True)
        for bad in [replace(o,quality=.5),replace(o,content_age_s=2),replace(o,durable=False),replace(o,finished_s=11)]:self.assertEqual(e.fraction([bad]),0)
    def test_population_includes_not_admitted(self):
        e=Evaluator([self.spec(),self.spec('b')]);self.assertEqual(e.fraction([Observed('a',1,'abc',0,1,True)]),.5)
    def test_estimate_no_future_truth(self):
        x=self.estimates();x['future_truth']={};self.assertRaises(ObservationDenied,validate_estimates,x)
        x=self.estimates();x['forecasts'][0]['realized_truth']={};self.assertRaises(ObservationDenied,validate_estimates,x)
    def test_common_observations_and_budget_mutations(self):
        p=[Policy('mpc'),Policy('hysteresis')];x=self.estimates();validate_pair(p,[x,x])
        y=copy.deepcopy(x);y['estimated_paths'][0]['up']=False;self.assertRaises(ContractError,validate_pair,p,[x,y]);self.assertRaises(ContractError,validate_pair,[p[0],replace(p[1],max_evaluations=1)],[x,x])
    def test_policy_family_baselines(self):
        for family in ['fallback','static','rule','greedy','hysteresis','fixed_target','mpc']:
            with self.subTest(family=family):
                d=decide(Policy(family,deadline_s=.1),self.estimates());self.assertNotIn(d.status,{'MODEL_FAILURE','BUDGET_EXCEEDED'});self.assertEqual(set(d.consumed_fields),set(FIELDS));self.assertLessEqual(d.model_evaluations,512)
    def test_empty_queue_defers(self):
        for family in ['fallback','static','rule','greedy','hysteresis','mpc']:self.assertEqual(decide(Policy(family),self.estimates(task_queue=[])).action,Action.defer())
    def test_estimated_catalog_resource_limits(self):
        x=self.estimates();x['task_queue']*=3
        for i,t in enumerate(x['task_queue']):x['task_queue'][i]=dict(t,task_id=str(i))
        h=Hardware()
        for a in catalog(x,h):
            if a.placement=='defer':continue
            e=Environment([self.spec(str(i)) for i in range(3)],100);self.assertEqual(validate_action(self.truth(e,cpu=10),a),())
    def test_stale_estimate_cannot_execute_unavailable(self):
        e=Environment([self.spec(local_eligible=False)],10);x=self.estimates();x['task_queue'][0]['local_eligible']=False;d=decide(Policy('fallback'),x);r=self.step(e,d.action,up=False);self.assertFalse(r['accepted']);self.assertEqual(e.evaluator.fraction(e.observed),0)
    def test_compute_budget_rejects_extra_calls(self):
        d=decide(Policy('mpc',max_evaluations=1),self.estimates());self.assertEqual(d.status,'BUDGET_EXCEEDED');self.assertEqual(d.model_evaluations,1);self.assertEqual(d.action,Action.defer())
    def test_whole_decision_elapsed_rejects(self):
        times=iter([0,0,.2,.2]);d=decide(Policy('greedy'),self.estimates(),clock=lambda:next(times));self.assertEqual(d.status,'BUDGET_EXCEEDED')
    def test_predictor_failure_defers(self):
        with patch('fair_control.controllers.predict',side_effect=ContractError('broken predictor')):
            d=decide(Policy('mpc'),self.estimates());self.assertEqual(d.status,'MODEL_FAILURE');self.assertEqual(d.action,Action.defer());self.assertIn('broken',d.error)
    def test_hysteresis_reuses_same_ranked_objective(self):
        x=self.estimates(previous_action=asdict(self.action(place='edge')));d=decide(Policy('hysteresis'),x);self.assertEqual(d.action.placement,'edge')
    def test_mpc_bounded_search_status(self):
        d=decide(Policy('mpc'),self.estimates());self.assertEqual(d.status,'BOUNDED_SEARCH_COMPLETE');self.assertIsNotNone(d.predicted_completions)
    def test_mpc_uses_predictions_not_future_truth(self):
        x=self.estimates();before=digest(x);d=decide(Policy('mpc'),x);self.assertEqual(digest(x),before);self.assertEqual(d.observation_sha256,before)
    def test_deterministic_controller_non_timing(self):
        x=self.estimates();a=asdict(decide(Policy('mpc'),x));b=asdict(decide(Policy('mpc'),x));a.pop('elapsed_s');b.pop('elapsed_s');self.assertEqual(a,b)
    def test_calibration_reference_known_math(self):
        b=b'constructed:1,3';c=Calibration('power','W','SYNTHETIC_ASSUMPTION','constructed:reference',hashlib.sha256(b).hexdigest(),'Original constructed fixture; rights pending','Authored exact reference','Not field uncertainty','Adapter check only','Uncalibrated synthetic',2.)
        s=reference_summary(c,[{'value':1,'units':'W'},{'value':3,'units':'W'}],b);self.assertEqual(s['mean'],2);self.assertAlmostEqual(s['sample_sd'],2**.5)
    def test_calibration_unit_and_hash_mutations(self):
        b=b'constructed';c=Calibration('power','W','SYNTHETIC_ASSUMPTION','constructed:reference',hashlib.sha256(b).hexdigest(),'Pending','Authored','None estimated','Engineering','Synthetic',1.)
        self.assertRaises(ContractError,reference_summary,c,[{'value':1,'units':'J'}],b);self.assertRaises(ContractError,reference_summary,c,[{'value':1,'units':'W'}],b'wrong');self.assertRaises(ContractError,replace,c,status='MEASURED')
    def test_calibration_status_no_claim(self):
        self.assertRaises(ContractError,Calibration,'power','W','MEASURED','unavailable',None,'pending','unknown','unknown','unknown','unknown',1)
    def config(self):return json.loads((ROOT/'fixtures/p5_offline.json').read_text())
    def test_runner_guard_research_modes(self):
        for mode in ['PILOT_EXPERIMENT','CONFIRMATORY_EXPERIMENT','HELDOUT_EVALUATION']:
            c=self.config();c['mode']=mode
            with tempfile.TemporaryDirectory() as p:
                self.assertRaises(ContractError,run,c,Path(p)/'out');self.assertFalse((Path(p)/'out').exists())
    def test_runner_authorization_mutation(self):
        c=self.config();c['authorization']['pilot_execution']=True;self.assertRaises(ContractError,validate_fixture,c)
    def test_no_holdout_path_or_seed(self):
        c=self.config();c['seed_manifest']['heldout_paths']=['future'];self.assertRaises(ContractError,validate_fixture,c)
        c=self.config();c['seed_manifest']['seed_ids']=[42];self.assertRaises(ContractError,validate_fixture,c)
    def test_runner_bounded_fixture_identity(self):
        c=self.config();c['tasks'][0]['work_mcycles']=99;self.assertRaises(ContractError,validate_fixture,c)
    def test_runner_deterministic_payload(self):
        a=run(self.config());b=run(self.config());self.assertEqual(a['payload_sha256'],b['payload_sha256']);self.assertEqual(a['payload']['primary_outcome']['successes'],1);self.assertFalse(a['payload']['campaign_executed'])
    def test_output_integrity_mutation(self):
        r=run(self.config());r['payload']['primary_outcome']['successes']=99;self.assertRaises(ContractError,validate_output,r)
    def test_source_manifest_mutation_and_completeness(self):
        m=tree_manifest();verify_manifest(m);m=copy.deepcopy(m);m['files'].pop(next(iter(m['files'])));self.assertRaises(ContractError,verify_manifest,m)
    def test_provenance_empty_revision_rejected(self):
        r=run(self.config());r['payload']['source_sha']='';r['payload_sha256']=digest(r['payload']);self.assertRaises(ContractError,validate_output,r)
