"""Constructed deterministic fixtures only, NOT a scientific comparison campaign."""
import unittest,math,copy,json
from pathlib import Path
from dataclasses import replace,FrozenInstanceError
from jsonschema import ValidationError,Draft202012Validator
from fair_control.core import *
from fair_control.analysis import paired_summary,plan_size
from fair_control.validation import ROOT,validate_config,validate_result,validate_future_result_shape,provenance

class Fixtures(unittest.TestCase):
    def setUp(self):
        self.truth=TruthFrame('fixture-only',('a','b'),(PathTruth('tn',True,10),PathTruth('ntn',True,3)),(('local',2.),('edge',20.),('cloud',30.)),Resources(10,20,100,5,50),2)
        self.action=Action('tn','edge',(Allocation('a',2,4,20,1),Allocation('b',2,4,20,1)))
    def test_truth_is_frozen(self):
        with self.assertRaises(FrozenInstanceError):self.truth.frame_id='changed'
        with self.assertRaises(ContractError):replace(self.truth,placement_cpu=(['edge',20],))
    def test_observation_does_not_retain_truth_or_alias_inputs(self):
        estimates={'estimated_paths':{'tn':{'capacity_mbps':4}},'future_truth':'secret'}
        v=Observation(estimates,['estimated_paths']);estimates['estimated_paths']['tn']['capacity_mbps']=0
        self.assertEqual(v['estimated_paths']['tn']['capacity_mbps'],4)
        with self.assertRaises(TypeError):v['estimated_paths']['tn']['capacity_mbps']=7
        with self.assertRaises(ObservationDenied):v['future_truth']
        self.assertEqual(v.consumed_fields,('estimated_paths','estimated_paths'))
    def test_same_primary_view_for_both_policies(self):
        c=json.loads((ROOT/'configs/design.json').read_text());est={k:None for k in c['policies'][0]['observations']}
        views=[Observation(est,p['observations']) for p in c['policies']]
        self.assertEqual(tuple(views[0]),tuple(views[1]))
        self.assertEqual(dict(views[0]),dict(views[1]))
    def test_valid_action_and_inclusive_resource_bound(self):
        self.assertEqual(validate_action(self.truth,self.action),())
        a=Action('tn','edge',(Allocation('a',10,20,100,5),))
        self.assertEqual(validate_action(self.truth,a),())
    def test_unavailable_gateway_visibility_or_terminal_power_rejected(self):
        for flag in ['up','visible','gateway_up','terminal_power']:
            p=replace(self.truth.paths[0],**{flag:False});t=replace(self.truth,paths=(p,self.truth.paths[1]))
            self.assertIn('unavailable_path',validate_action(t,self.action))
    def test_each_resource_budget_is_enforced(self):
        for field,value in [('bandwidth_mbps',10.000001),('cpu_mcycles_s',20.000001),('memory_mb',100.000001),('power_w',5.000001)]:
            a=Action('tn','edge',(replace(self.action.allocations[0],**{field:value}),))
            self.assertIn('over_'+field,validate_action(self.truth,a))
    def test_capacities_not_merely_positive_flags(self):
        a=replace(self.action,network='ntn')
        self.assertIn('path_capacity_exceeded',validate_action(self.truth,a))
        t=replace(self.truth,placement_cpu=(('edge',1.),))
        self.assertIn('placement_cpu_exceeded',validate_action(t,self.action))
    def test_transition_energy_included(self):
        self.assertIn('energy_exceeded',validate_action(self.truth,self.action,cost=TransitionCost(0,47)))
    def test_nonfinite_and_negative_allocations_fail(self):
        for v in [-1,float('nan'),float('inf')]:
            a=replace(self.action,allocations=(replace(self.action.allocations[0],power_w=v),))
            self.assertIn('invalid_power_w',validate_action(self.truth,a))
    def test_duplicate_unknown_or_invalid_task_allocations_fail(self):
        self.assertIn('duplicate_task',validate_action(self.truth,replace(self.action,allocations=(self.action.allocations[0],)*2)))
        self.assertIn('unknown_task',validate_action(self.truth,replace(self.action,allocations=(replace(self.action.allocations[0],task_id='unknown'),))))
        self.assertIn('invalid_allocation_contract',validate_action(self.truth,replace(self.action,allocations=('bad',))))
    def test_defer_is_explicit_zero_resource_action(self):
        self.assertEqual(validate_action(self.truth,Action.defer()),())
        self.assertIn('defer_must_use_no_resource',validate_action(self.truth,Action('tn','defer')))
    def test_local_execution_cannot_claim_network_resources(self):
        a=Action(None,'local',(Allocation('a',0,1,10,.5),))
        self.assertEqual(validate_action(self.truth,a),())
        self.assertIn('local_bandwidth_nonzero',validate_action(self.truth,replace(a,allocations=(replace(a.allocations[0],bandwidth_mbps=1),))))
    def test_first_action_no_switch_and_common_cost(self):
        a=replace(self.action,network='ntn',placement='cloud')
        costs={'network':TransitionCost(.2,2),'placement':TransitionCost(.4,3)}
        self.assertEqual(transition(None,a,**costs),TransitionCost())
        self.assertEqual(transition(self.action,a,**costs),TransitionCost(.6000000000000001,5))
        self.assertIn('transition_consumes_interval',validate_action(self.truth,self.action,cost=TransitionCost(2,0)))
    def test_compute_limit_counts_actual_calls_and_timeout_defers(self):
        calls=[];ledger=ComputeLedger(2,1,clock=lambda:0)
        a=choose([self.action]*3,lambda x:calls.append(x) or 1,ledger)
        self.assertEqual(a,Action.defer());self.assertEqual(len(calls),2);self.assertEqual(ledger.evaluations,2);self.assertTrue(ledger.exceeded)
    def test_elapsed_time_budget_includes_scorer_time(self):
        now=[0.];ledger=ComputeLedger(5,.1,clock=lambda:now[0])
        def slow(a):now[0]=.2;return 1
        self.assertEqual(choose([self.action],slow,ledger),Action.defer())
        self.assertTrue(ledger.exceeded)
    def test_common_objective_hook_and_stable_tie_order(self):
        other=replace(self.action,network='ntn');ledger=lambda:ComputeLedger(5,1,clock=lambda:0)
        self.assertEqual(choose([self.action,other],lambda a:1 if a.network=='ntn' else 0,ledger()),other)
        self.assertEqual(choose([self.action,other],lambda a:1,ledger()),self.action)
    def test_missing_tasks_stay_in_denominator(self):
        req=[TaskRequirement(k,0,1,1,.9) for k in ['a','b']];out=[TaskOutcome('a',.5,0,1,True,True)]
        self.assertEqual(completion_fraction(req,out),.5)
    def test_connected_but_wrong_stale_late_poor_or_nondurable_fails(self):
        req=[TaskRequirement('a',0,1,1,.9)];good=TaskOutcome('a',.5,0,1,True,True)
        self.assertEqual(completion_fraction(req,[good]),1)
        for changes in [{'correct':False},{'content_age_s':2},{'finished_s':2},{'quality':.5},{'durable':False}]:
            self.assertEqual(completion_fraction(req,[replace(good,**changes)]),0)
    def test_duplicate_outcomes_cannot_inflate_success(self):
        req=[TaskRequirement('a',0,1,1,.9)];o=TaskOutcome('a',.5,0,1,True,True)
        with self.assertRaises(ContractError):completion_fraction(req,[o,o])
    def test_seed_pairing_uses_ids_not_insertion_order(self):
        a={'x':.4,'y':.8,'z':.6};b={'z':.3,'x':.1,'y':.5}
        r=paired_summary(a,b);self.assertAlmostEqual(r['mean'],.3);self.assertEqual(r['unit_ids'],['x','y','z'])
        with self.assertRaises(ContractError):paired_summary(a,{'wrong':.2})
    def test_statistics_sd_ci_and_zero_not_equivalence(self):
        a={'x':.2,'y':.6,'z':1.};b={k:.5 for k in a};r=paired_summary(a,b)
        self.assertAlmostEqual(r['mean'],.1);self.assertAlmostEqual(r['sample_sd'],.4)
        self.assertLess(r['ci95'][0],0);self.assertIn('not equivalence',r['interpretation'])
        self.assertIsNone(paired_summary({'a':.5,'b':.5},{'a':.5,'b':.5})['paired_dz'])
    def test_precision_plan_hypothetical_variance_and_cap(self):
        r=plan_size(.15);self.assertTrue(r['feasible_within_cap']);self.assertGreaterEqual(r['n'],865);self.assertFalse(r['heldout_may_start'])
        self.assertFalse(plan_size(1,cap=2500)['feasible_within_cap'])
    def test_config_valid_draft_and_invalid_mutations(self):
        c=json.loads((ROOT/'configs/design.json').read_text());validate_config(c)
        mutations=[lambda x:x['authorization'].update(campaign_execution=True),lambda x:x['heldout'].update(seed_ids=[9]),lambda x:x['policies'][1].update(observations=['future_truth']),lambda x:x['compute'].update(deadline_s=float('nan')),lambda x:x.update(unknown=True)]
        for mutation in mutations:
            b=copy.deepcopy(c);mutation(b)
            with self.assertRaises((ContractError,ValidationError,ValueError)):validate_config(b)
    def test_canonical_hash_detects_scientific_configuration_change(self):
        a={'cost':.2,'units':'seconds'};self.assertEqual(digest(a),digest({'units':'seconds','cost':.2}));self.assertNotEqual(digest(a),digest({**a,'cost':.3}))
        with self.assertRaises(ValueError):canonical({'cost':float('nan')})
    def test_schemas_are_valid_and_report_cannot_claim_campaign(self):
        for p in (ROOT/'schemas').glob('*.json'):Draft202012Validator.check_schema(json.loads(p.read_text()))
        record={'schema_version':'0.1','evidence_class':'ENGINEERING_VALIDATION','campaign_executed':False,'publication_authorized':False,'tests_run':1,'failures':0,'errors':0,'source_tree_sha256':'0'*64};validate_result(record)
        record['campaign_executed']=True
        with self.assertRaises(ValidationError):validate_result(record)
    def test_raw_truth_rejects_invalid_probability_flags_and_numbers(self):
        with self.assertRaises(ContractError):PathTruth('tn',1,1)
        with self.assertRaises(ContractError):Resources(1,1,float('inf'),1,1)
        with self.assertRaises(ContractError):Observation({'x':set([1])},['x'])

    def test_observation_selection_cannot_guarantee_true_feasibility(self):
        # Constructed stale estimate: same positive estimate can lead both policies
        # to an action rejected by the common execution truth. No truth is passed
        # through the policy callback, and missing task outcomes remain failures.
        obs=Observation({'estimated_paths':{'tn':{'up':True}}},['estimated_paths'])
        selected=choose([self.action],lambda a:float(obs['estimated_paths']['tn']['up']),ComputeLedger(1,1,clock=lambda:0))
        truth=replace(self.truth,paths=(replace(self.truth.paths[0],up=False),self.truth.paths[1]))
        self.assertIn('unavailable_path',validate_action(truth,selected))
        self.assertEqual(completion_fraction([TaskRequirement('a',0,1,1,.9)],[]),0)
    def test_future_result_population_and_nan_rejected(self):
        # Schema fixture only, never a populated research output.
        r={'source_sha':'0'*40,'config_sha256':'0'*64,'input_sha256':'0'*64,'unit_id':'constructed-only','policy_id':'constructed-only','scheduled_tasks':1,'successes':1,'invalid_actions':0,'timeouts':0,'model_evaluations':1,'decision_seconds':.01,'observation_digest':'0'*64,'action_trace_sha256':'0'*64,'outcome_trace_sha256':'0'*64,'evidence_class':'PROSPECTIVE_DIGITAL_SIMULATION'}
        validate_future_result_shape(r)
        with self.assertRaises(ContractError):validate_future_result_shape({**r,'successes':2})
        with self.assertRaises(ValueError):validate_future_result_shape({**r,'decision_seconds':float('nan')})
