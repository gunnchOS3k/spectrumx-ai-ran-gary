"""P6 constructed engineering/adversarial checks; no prospective inference."""
import unittest,tempfile,json,copy,hashlib,math
from pathlib import Path
from dataclasses import replace,asdict
from unittest.mock import patch
from fair_control.core import ContractError,canonical,digest,Action
from fair_control.p6_application import catalog,catalog_tasks,run_reference,evaluate,ReferenceResult,TEXT,NOTE,bytes_hash
from fair_control.p6_scenarios import Gate,Link,NetworkProcess,CausalEstimator,FluidQueue,poisson_count,burst_count
from fair_control.p6_calibration import CalibrationRecord,inspect_reference,convert
from fair_control.p6_statistics import planning,estimand
from fair_control.p6_configuration import candidate,validate_candidate,verify_seal,execution_guard,AUTH,ROOT

class P6(unittest.TestCase):
    def task(self,id):return next(t for t in catalog_tasks() if t.task_id==id)
    def reference(self,id,**kw):
        with tempfile.TemporaryDirectory() as p:
            task=self.task(id);r=run_reference(task,Path(p),**kw);return r,evaluate(task,r)
    def config(self):return candidate('aa0f1d28a94a0e9058bf4f5b85ca99d592c00561',hashlib.sha256((ROOT.parent/'PROSPECTIVE_PROTOCOL.md').read_bytes()).hexdigest())
    def network(self):
        return NetworkProcess([Gate('power',1,1),Gate('TN',1,1),Gate('NTN',1,1)],[Link('TN',10,.02,('power','TN')),Link('NTN',3,.2,('power','NTN'))])
    def test_known_program_exact_output(self):
        r,v=self.reference('known-program');self.assertEqual(r.output_utf8,'55\n');self.assertTrue(v['success'])
    def test_program_corruption_independently_rejected(self):
        with patch('fair_control.p6_application.PROGRAM','print(56)\n'):
            r,v=self.reference('known-program');self.assertFalse(v['success'])
    def test_cached_lesson_progress_durable_offline(self):
        r,v=self.reference('cached-lesson',network_up=False);self.assertTrue(v['success']);self.assertEqual(r.persisted_utf8,TEXT)
    def test_corrupt_content(self):
        r,v=self.reference('cached-lesson',content_utf8='corrupt');self.assertEqual(r.failure,'corrupt_content');self.assertFalse(v['success'])
    def test_stale_content_rejected(self):
        r,v=self.reference('cached-lesson',virtual_now_s=31);self.assertFalse(v['success'])
    def test_document_edit_reopen(self):
        r,v=self.reference('local-document');self.assertEqual(r.output_utf8,NOTE);self.assertEqual(r.persisted_utf8,NOTE);self.assertTrue(v['success'])
    def test_restore_reference(self):
        r,v=self.reference('restore-note');self.assertTrue(v['success']);self.assertEqual(r.persisted_utf8,NOTE)
    def test_corrupt_restore_reference(self):
        r,v=self.reference('restore-note',corrupt_checkpoint=True);self.assertFalse(v['success']);self.assertEqual(r.failure,'checkpoint_integrity')
    def test_sync_requires_network_and_deduplicates(self):
        r,v=self.reference('idempotent-sync',network_up=False);self.assertFalse(v['success']);r,v=self.reference('idempotent-sync',network_up=True);self.assertTrue(v['success']);self.assertEqual(json.loads(r.output_utf8)['applied'],1)
    def test_unsupported_reference(self):
        r,v=self.reference('unsupported-server',network_up=True);self.assertFalse(v['success']);self.assertEqual(r.failure,'unsupported_workload')
    def test_reference_wrong_durable_quality_deadline(self):
        r,v=self.reference('local-document');t=self.task('local-document')
        for bad in [replace(r,persisted_utf8='wrong'),replace(r,quality=.5),replace(r,virtual_finished_s=6),replace(r,output_utf8='wrong')]:self.assertFalse(evaluate(t,bad)['success'])
    def test_reference_future_content_forbidden(self):
        r,v=self.reference('local-document');self.assertRaises(ContractError,evaluate,self.task('local-document'),replace(r,generated_s=2))
    def test_nonempty_workspace_forbidden(self):
        with tempfile.TemporaryDirectory() as p:
            (Path(p)/'unrelated').write_text('keep');self.assertRaises(ContractError,run_reference,self.task('known-program'),Path(p))
    def test_task_catalog_hash_and_approval(self):
        self.assertEqual(len(catalog()['tasks']),6);self.assertFalse(catalog()['owner_scientific_approval']);self.assertEqual(digest(catalog()),digest(json.loads(canonical(catalog()))))
    def test_common_power_failure_both_paths(self):
        p=self.network();f=p.advance({'power':0,'TN':.99,'NTN':.99});self.assertTrue(all(not x['up'] for x in f['paths']))
    def test_independent_component_failure_one_path(self):
        p=self.network();f=p.advance({'power':.99,'TN':0,'NTN':.99});self.assertEqual([x['up'] for x in f['paths']],[False,True])
    def test_network_recovery(self):
        p=self.network();p.advance({'power':0,'TN':.99,'NTN':.99});f=p.advance({'power':0,'TN':.99,'NTN':.99});self.assertTrue(all(x['up'] for x in f['paths']))
    def test_no_zero_rate_failure(self):
        p=NetworkProcess([Gate('x',0,0)],[Link('TN',1,.1,('x',))]);self.assertTrue(p.advance({'x':0})['paths'][0]['up'])
    def test_rate_duration_math(self):
        p=NetworkProcess([Gate('x',math.log(2),0)],[Link('TN',1,0,('x',))]);self.assertFalse(p.advance({'x':.49})['paths'][0]['up']);q=NetworkProcess([Gate('x',math.log(2),0)],[Link('TN',1,0,('x',))]);self.assertTrue(q.advance({'x':.51})['paths'][0]['up'])
    def test_outage_inputs_mutations(self):
        p=self.network()
        for x in [{'power':0},{'power':1,'TN':.2,'NTN':.3},{'power':float('nan'),'TN':.2,'NTN':.3}]:self.assertRaises(ContractError,p.advance,x)
        self.assertRaises(ContractError,Link,'TN',1,0,('x','x'))
    def test_background_capacity_enforced(self):
        p=self.network();f=p.advance({'power':.99,'TN':.99,'NTN':.99},{'TN':12});self.assertEqual(f['paths'][0]['capacity_mbps'],0)
    def test_fluid_queue_conserves_data(self):
        q=FluidQueue();r=q.advance(12,10,1);self.assertEqual(r['served_mbit'],10);self.assertEqual(r['backlog_mbit'],2);self.assertEqual(r['queue_delay_s'],.2);r=q.advance(0,1,1);self.assertEqual(r['backlog_mbit'],1)
    def test_queue_invalid_and_zero_capacity(self):
        q=FluidQueue();self.assertRaises(ContractError,q.advance,-1,1,1);r=q.advance(2,0,1);self.assertEqual(r['backlog_mbit'],2);self.assertIsNone(r['queue_delay_s'])
    def test_poisson_inverse_known_counts(self):
        self.assertEqual(poisson_count(0,1,.99),0);self.assertEqual(poisson_count(1,1,.1),0);self.assertEqual(poisson_count(1,1,.5),1);self.assertEqual(poisson_count(1,1,.8),2)
    def test_burst_and_arrival_bounds(self):
        self.assertEqual(burst_count(3),3);self.assertRaises(ContractError,burst_count,65);self.assertRaises(ContractError,poisson_count,51,1,.5);self.assertRaises(ContractError,poisson_count,1,1,1)
    def test_network_restart_replay(self):
        p=self.network();p.advance({'power':.99,'TN':0,'NTN':.99});q=self.network();q.restore(json.loads(canonical(p.snapshot())));self.assertEqual(p.advance({'power':.99,'TN':0,'NTN':.99}),q.advance({'power':.99,'TN':0,'NTN':.99}))
    def test_estimator_delayed_observations(self):
        p=self.network();e=CausalEstimator(delay_s=1);a=p.advance({'power':.99,'TN':.99,'NTN':.99});e.ingest(a,0);self.assertRaises(ContractError,e.observe,0)
        b=p.advance({'power':.99,'TN':0,'NTN':.99});e.ingest(b,1);o=e.observe(1);self.assertTrue(o['estimated_paths'][0]['up']);self.assertFalse(b['paths'][0]['up']);o=e.observe(2);self.assertFalse(o['estimated_paths'][0]['up'])
    def test_estimator_error_does_not_change_truth(self):
        p=self.network();f=p.advance({'power':.99,'TN':.99,'NTN':.99});h=digest(f);e=CausalEstimator(capacity_relative_error=-.2,latency_bias_s=.1,flip_up=True);e.ingest(f,0);o=e.observe(0);self.assertEqual(digest(f),h);self.assertEqual(o['estimated_paths'][0]['capacity_mbps'],8);self.assertFalse(o['estimated_paths'][0]['up'])
    def test_forecasts_persist_history_not_future(self):
        p=self.network();f=p.advance({'power':.99,'TN':.99,'NTN':.99});e=CausalEstimator();e.ingest(f,0);o=e.observe(0);p.advance({'power':0,'TN':.99,'NTN':.99});self.assertTrue(o['forecasts'][0]['paths'][0]['up']);self.assertFalse(o['provenance']['future_truth_access'])
    def test_late_delivery_not_available_in_past(self):
        p=self.network();f=p.advance({'power':.99,'TN':.99,'NTN':.99});e=CausalEstimator();e.ingest(f,10);self.assertRaises(ContractError,e.observe,1);self.assertTrue(e.observe(10)['estimated_paths'][0]['up'])
    def test_future_frame_ingest_rejected(self):
        p=self.network();f=p.advance({'power':.99,'TN':.99,'NTN':.99});f['timestamp_s']=2;e=CausalEstimator();self.assertRaises(ContractError,e.ingest,f,1)
    def test_estimator_restart(self):
        p=self.network();e=CausalEstimator();e.ingest(p.advance({'power':.99,'TN':.99,'NTN':.99}),0);q=CausalEstimator.restart(json.loads(canonical(e.snapshot())));self.assertEqual(e.observe(1),q.observe(1))
    def test_calibration_unit_conversion(self):
        self.assertEqual(convert(100,'ms','s'),.1);self.assertEqual(convert(1,'MiB','MB'),1.048576);self.assertRaises(ContractError,convert,100,'ms','W')
    def record(self):
        b=b'constructed reference';r=CalibrationRecord('latency','SYNTHETIC_ASSUMPTION','s','constructed:unit reference',hashlib.sha256(b).hexdigest(),'Authored exact reference','Not field uncertainty','Adapter unit test','No physical transfer','Original fixture','LOCAL_USE_VERIFIED',False,.1);return r,b
    def test_calibration_constructed_reference(self):
        r,b=self.record();x=inspect_reference(r,[{'value':100,'units':'ms','metric_definition':'latency'},{'value':300,'units':'ms','metric_definition':'latency'}],b);self.assertAlmostEqual(x['mean'],.2);self.assertFalse(x['physical_calibration_established'])
    def test_calibration_mutations(self):
        r,b=self.record();self.assertRaises(ContractError,replace,r,status='MEASURED');self.assertRaises(ContractError,replace,r,imported=True,rights_review_status='PENDING');self.assertRaises(ContractError,inspect_reference,r,[{'value':1,'units':'W','metric_definition':'latency'}],b);self.assertRaises(ContractError,inspect_reference,r,[{'value':100,'units':'ms','metric_definition':'latency'}],b'wrong')
    def test_planning_normal_assumption_boundaries(self):
        p=planning(.15);self.assertGreater(p['n_required_within_cap'],800);self.assertLess(p['n_required_within_cap'],1000);self.assertGreater(p['pilot_half_width_if_sd_holds'],.04);self.assertLess(p['pilot_power_if_sd_holds'],.2);self.assertFalse(p['pilot_execution_authorized'])
    def test_planning_cap_and_alternative(self):
        self.assertIsNone(planning(.3)['n_required_within_cap']);self.assertRaises(ContractError,planning,.15,alternative=.02);self.assertRaises(ContractError,planning,float('nan'))
    def test_estimand_weights_are_distinct(self):
        e=estimand([1,0],[0,0],[1,9]);self.assertEqual(e['equal_unit_mean'],.5);self.assertEqual(e['task_weighted_difference'],.1)
    def test_estimand_invalid_denominator(self):
        self.assertRaises(ContractError,estimand,[2],[0],[1]);self.assertRaises(ContractError,estimand,[0],[0],[0])
    def test_candidate_complete_valid_unapproved(self):
        c=self.config();x=validate_candidate(c);self.assertTrue(x['configuration_technically_validated']);self.assertFalse(x['scientific_design_approved']);self.assertFalse(x['pilot_execution_authorized'])
    def test_configuration_missing_and_numeric_mutation(self):
        c=self.config();c.pop('stop_conditions');self.assertRaises(ContractError,validate_candidate,c)
        c=self.config();c['hardware']['memory_mb']=-1;self.assertRaises(ContractError,validate_candidate,c)
        c=self.config();c['parameters']['hysteresis']['value']=float('nan');self.assertRaises((ContractError,ValueError),validate_candidate,c)
    def test_configuration_scientific_incompatibilities(self):
        c=self.config();c['parameters']['control_interval']['value']=2;self.assertRaises(ContractError,validate_candidate,c)
        c=self.config();c['parameters']['margin']['value']=.5;self.assertRaises(ContractError,validate_candidate,c)
        c=self.config();c['parameters']['memory']['value']=101;self.assertRaises(ContractError,validate_candidate,c)
    def test_primary_equal_budget_mutation(self):
        c=self.config();c['policies'][1]['max_evaluations']=500;self.assertRaises(ContractError,validate_candidate,c)
    def test_configuration_no_implicit_approval(self):
        c=self.config();c['authorization']['pilot_execution_authorized']=True;self.assertRaises(ContractError,validate_candidate,c)
        c=self.config();c['parameters']['margin']['owner_approved']=True;self.assertRaises(ContractError,validate_candidate,c)
    def test_configuration_stage_leakage(self):
        c=self.config();c['analysis']['pilot_reused_confirmatory']=True;self.assertRaises(ContractError,validate_candidate,c)
        c=self.config();c['holdout_access']['outcome_paths']=['reserved'];self.assertRaises(ContractError,validate_candidate,c)
    def test_source_revision_mutation(self):
        c=self.config();c['source']['modules']['controllers.py']='0'*64;self.assertRaises(ContractError,validate_candidate,c)
    def test_seal_detects_rehashed_scientific_config(self):
        c=self.config();seal={'candidate_sha256':digest(c),'source':c['source'],'authorization':AUTH};verify_seal(c,seal)
        c['parameters']['initial_energy']['value']=200;self.assertRaises(ContractError,verify_seal,c,seal)
    def test_disabled_modes_no_output(self):
        for mode in ['PILOT_EXPERIMENT','CONFIRMATORY_EXPERIMENT','HELDOUT_EVALUATION','ENGINEERING_TEST']:
            with tempfile.TemporaryDirectory() as p:
                dest=Path(p)/'not_created';self.assertRaises(ContractError,execution_guard,mode,None,dest);self.assertFalse(dest.exists())
    def view(self):
        return {'timestamp_s':0,'estimated_paths':[{'name':'TN','up':False,'capacity_mbps':1,'latency_s':0}],'task_queue':[],'remaining_energy_j':1,'previous_action':None,'forecasts':[]}
    def test_process_accounting_and_reap(self):
        from fair_control.p6_process import controlled_decision
        from fair_control.controllers import Policy
        d,x=controlled_decision(Policy('hysteresis'),self.view());self.assertEqual(x['process_status'],'RETURNED');self.assertGreaterEqual(x['startup_s'],0);self.assertGreaterEqual(x['worker_decision_s'],0);self.assertGreaterEqual(x['decision_window_s'],x['worker_decision_s']);self.assertTrue(x['direct_child_reaped']);self.assertGreaterEqual(x['child_cpu_s'],0);self.assertGreaterEqual(x['parent_cpu_s'],0)
    def test_child_decision_timeout(self):
        from fair_control.p6_process import controlled_decision
        from fair_control.controllers import Policy
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'hang.py';f.write_text("import time\nprint('READY',flush=True)\ntime.sleep(5)\n");d,x=controlled_decision(Policy('greedy'),self.view(),worker=f);self.assertEqual(d.action,Action.defer());self.assertEqual(x['process_status'],'DECISION_TIMEOUT');self.assertTrue(x['direct_child_reaped'])
    def test_child_startup_timeout(self):
        from fair_control.p6_process import controlled_decision
        from fair_control.controllers import Policy
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'hang.py';f.write_text('import time\ntime.sleep(5)\n');d,x=controlled_decision(Policy('greedy'),self.view(),startup_allowance_s=.05,worker=f);self.assertEqual(d.action,Action.defer());self.assertEqual(x['process_status'],'STARTUP_TIMEOUT_OR_ERROR')

    def test_duration_aggregate_adapter_correct_math(self):
        from fair_control.p6_calibration import azure_duration_adapter
        x=azure_duration_adapter([{'Average':100,'Count':2},{'Average':400,'Count':1}]);self.assertAlmostEqual(x['weighted_mean_duration_s'],.2);self.assertFalse(x['cold_start_included']);self.assertFalse(x['raw_invocation_variance_available'])
    def test_duration_aggregate_not_percentile_or_counts(self):
        from fair_control.p6_calibration import azure_duration_adapter
        self.assertRaises(ContractError,azure_duration_adapter,[{'Average':100,'Count':0}]);self.assertRaises(ContractError,azure_duration_adapter,[{'percentile':100,'Count':1}])
    def test_latency_semantics_not_rtt(self):
        from fair_control.p6_calibration import one_way_latency_adapter
        self.assertEqual(one_way_latency_adapter([{'value':20,'units':'ms','semantics':'ONE_WAY_END_TO_END'}]),[.02]);self.assertRaises(ContractError,one_way_latency_adapter,[{'value':20,'units':'ms','semantics':'RTT'}])
    def test_approval_mutation_does_not_change_other_candidate(self):
        c=self.config();c['authorization']['pilot_execution_authorized']=True;self.assertFalse(self.config()['authorization']['pilot_execution_authorized']);self.assertRaises(ContractError,validate_candidate,c)

    def test_application_bridge_verifies_real_reference_bytes(self):
        from fair_control.p6_bridge import verify_modeled_completion,application_fraction
        from fair_control.core import TaskOutcome
        t=self.task('known-program');o=TaskOutcome(t.task_id,1,0,1,True,True)
        with tempfile.TemporaryDirectory() as p:j=verify_modeled_completion(t,o,Path(p),network_up=False)
        self.assertTrue(j['success']);self.assertEqual(application_fraction([t],[j]),1)
        with patch('fair_control.p6_application.PROGRAM','print(56)\n'):
            with tempfile.TemporaryDirectory() as p:j=verify_modeled_completion(t,o,Path(p))
        self.assertFalse(j['success']);self.assertEqual(application_fraction([t],[j]),0)
    def test_bridge_keeps_missing_tasks_denominator(self):
        from fair_control.p6_bridge import application_fraction
        self.assertEqual(application_fraction(catalog_tasks(),[]),0)

    def test_causal_outage_then_recovery_application_completion(self):
        from fair_control.core import TruthFrame,PathTruth,Resources,Allocation,TaskRequirement
        from fair_control.tasks import TaskSpec,sha
        from fair_control.dynamics import Environment
        p=self.network();env=Environment([TaskSpec(TaskRequirement('a',0,2,10,1),4,1,'abc',sha('abc'),local_eligible=False)],10)
        action=Action('TN','edge',(Allocation('a',0,5,1,1),))
        for uniforms in [{'power':0,'TN':.99,'NTN':.99},{'power':0,'TN':.99,'NTN':.99}]:
            row=p.advance(uniforms);frame=TruthFrame('current',('a',),tuple(PathTruth(x['name'],x['up'],x['capacity_mbps']) for x in row['paths']),(('edge',10.),),Resources(10,10,100,2,10),1)
            env.step(frame,action,{x['name']:x['latency_s'] for x in row['paths']})
        self.assertFalse(env.trace[0]['accepted']);self.assertTrue(env.trace[1]['accepted']);self.assertEqual(env.evaluator.fraction(env.attempts),1)
    def test_protocol_revision_mutation(self):
        c=self.config();c['protocol_revision']['sha256']='0'*64;self.assertRaises(ContractError,validate_candidate,c)

    def test_bridge_relative_deadline_for_later_arrival(self):
        from fair_control.p6_bridge import verify_modeled_completion,application_fraction
        from fair_control.core import TaskOutcome
        t=self.task('known-program');o=TaskOutcome(t.task_id,11,1,1,True,True)
        with tempfile.TemporaryDirectory() as p:j=verify_modeled_completion(t,o,Path(p),released_s=10)
        self.assertTrue(j['success']);self.assertEqual(application_fraction([t],[j],{t.task_id:10}),1)

    def test_large_request_stalled_child_does_not_block_writer(self):
        from fair_control.p6_process import controlled_decision
        from fair_control.controllers import Policy
        x=self.view();x['task_queue']=[{'task_id':'x'*32000,'remaining_mcycles':1.,'remaining_mbit':0.,'memory_mb':1.,'deadline_s':5.,'minimum_quality':1.,'content_age_s':0.,'max_age_s':30.,'local_eligible':True,'network_required':False,'work_mcycles':1.,'input_mbit':0.,'placement':None,'network':None,'latency_paid_s':0.,'content_available':True,'supported':True}]
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'hang.py';f.write_text("import time\nprint('READY',flush=True)\ntime.sleep(5)\n");d,r=controlled_decision(Policy('greedy'),x,worker=f);self.assertEqual(d.action,Action.defer());self.assertEqual(r['process_status'],'DECISION_TIMEOUT');self.assertLess(r['outer_elapsed_s'],2)
