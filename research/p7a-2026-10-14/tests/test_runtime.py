import copy,json,tempfile,unittest
from pathlib import Path
from dataclasses import asdict
from p7a.session import Session,VERSION
from p7a.fixtures import fixtures
from fair_control.controllers import Policy
from fair_control.p6_control import decide_primary
from fair_control.core import ContractError,digest,Action

class Runtime(unittest.TestCase):
    def view(self):return fixtures()[1]['estimated_input']
    def worker(self,p,body):
        f=Path(p)/'worker.py';f.write_text("import sys,time,json\nprint(json.dumps({'ready':"+repr(VERSION)+"}),flush=True)\n"+body);return f
    def test_all_fixture_contracts(self):self.assertEqual([len(x['estimated_input']['task_queue']) for x in fixtures()],[0,1,8,64,8,64])
    def test_reused_one_step_small_matches_direct(self):
        v=self.view();p=Policy('hysteresis');expected=decide_primary(p,v)
        with Session(p) as s:
            d,r=s.decide(v);self.assertEqual(d.action,expected.action);self.assertEqual(d.model_evaluations,expected.model_evaluations);self.assertEqual(r['process_status'],'RETURNED');self.assertFalse(r['session_invalidated_on_failure'])
        self.assertTrue(s.summary['direct_child_reaped'])
    def test_reused_mpc_small_matches_direct(self):
        p=Policy('mpc');v=self.view();a=decide_primary(p,v)
        with Session(p) as s:
            b,r=s.decide(v);self.assertEqual(a.action,b.action);self.assertEqual(a.model_evaluations,b.model_evaluations);self.assertEqual(r['equal_max_model_calls'],512)
    def test_aba_no_cross_request_state(self):
        a=self.view();b=fixtures()[0]['estimated_input']
        with Session(Policy('hysteresis')) as s:
            d1,r1=s.decide(a);s.decide(b);d2,r2=s.decide(a)
        self.assertEqual(d1.action,d2.action);self.assertEqual(d1.observation_sha256,d2.observation_sha256);self.assertEqual(d1.model_evaluations,d2.model_evaluations);self.assertEqual([r1['request_id'],r2['request_id']],[0,2])
    def test_input_unchanged_and_family_contract(self):
        a=self.view();h=digest(a)
        with Session(Policy('mpc')) as s:d,r=s.decide(a)
        self.assertEqual(h,digest(a));self.assertEqual(d.action.fidelity,'target');self.assertEqual(r['virtual_service_debit_s'],0)
        self.assertRaises(ContractError,Session,Policy('greedy'))
    def test_unrequested_truth_rejected_before_send(self):
        v=self.view();v['future_truth']={}
        with Session(Policy('hysteresis')) as s:
            self.assertRaises(ContractError,s.decide,v);self.assertEqual(s.next_id,0)
    def test_65_visible_tasks_reject_not_trim(self):
        v=fixtures()[3]['estimated_input'];v['task_queue'].append({**v['task_queue'][0],'task_id':'extra'})
        with Session(Policy('hysteresis')) as s:self.assertRaises(ContractError,s.decide,v)
    def test_request_count_exhaustion(self):
        s=Session(Policy('hysteresis'),max_requests=1)
        try:s.decide(self.view());self.assertRaises(ContractError,s.decide,self.view())
        finally:s.close()
        self.assertRaises(ContractError,s.decide,self.view())
    def test_timeout_invalidates_reuse_and_reaps(self):
        with tempfile.TemporaryDirectory() as p:
            f=self.worker(p,"sys.stdin.readline()\ntime.sleep(2)\n")
            s=Session(Policy('hysteresis',deadline_s=.02),worker=f);d,r=s.decide(self.view())
            self.assertEqual(d.action,Action.defer());self.assertEqual(r['process_status'],'DECISION_TIMEOUT');self.assertIsNone(r['actual_model_evaluations']);self.assertTrue(s.summary['direct_child_reaped']);self.assertRaises(ContractError,s.decide,self.view())
    def test_partial_line_never_blocks_past_window(self):
        with tempfile.TemporaryDirectory() as p:
            f=self.worker(p,"sys.stdin.readline()\nsys.stdout.write('{');sys.stdout.flush()\ntime.sleep(2)\n")
            with Session(Policy('hysteresis',deadline_s=.02),worker=f) as s:
                d,r=s.decide(self.view());self.assertEqual(r['process_status'],'DECISION_TIMEOUT');self.assertTrue(s.closed)
    def test_wrong_reply_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as p:
            f=self.worker(p,"sys.stdin.readline()\nprint('{}',flush=True)\n")
            with Session(Policy('hysteresis'),worker=f) as s:
                d,r=s.decide(self.view());self.assertEqual(r['process_status'],'PROCESS_ERROR');self.assertEqual(d.action,Action.defer())
    def test_startup_failure_reaped(self):
        with tempfile.TemporaryDirectory() as p:
            f=Path(p)/'bad.py';f.write_text("raise RuntimeError('constructed error')\n");self.assertRaises(ContractError,Session,Policy('hysteresis'),worker=f)
    def test_equal_common_catalog_from_same_input(self):
        from fair_control.p6_control import primary_catalog
        from fair_control.controllers import Hardware
        from fair_control.dynamics import Model
        a=self.view();m=primary_catalog(a,Hardware(),Model());h=digest([asdict(x) for x in m])
        with Session(Policy('mpc')) as s:d,_=s.decide(a)
        with Session(Policy('hysteresis')) as s:b,_=s.decide(a)
        self.assertEqual(d.catalog_sha256,h);self.assertEqual(b.catalog_sha256,h)
    def test_budget_exhaustion_is_retained(self):
        with Session(Policy('mpc',max_evaluations=1)) as s:
            d,r=s.decide(self.view());self.assertEqual(d.status,'BUDGET_EXCEEDED');self.assertEqual(d.action,Action.defer());self.assertEqual(r['actual_model_evaluations'],1)
    def test_current_rss_bytes_if_supported(self):
        import os,sys
        from p7a.memory import resident_bytes
        x=resident_bytes(os.getpid())
        if sys.platform=='darwin':self.assertIsInstance(x,int);self.assertGreater(x,0)
        else:self.assertIsNone(x)
    def test_all_research_modes_guard_before_config_or_output(self):
        import subprocess,sys
        from p7a import ROOT
        with tempfile.TemporaryDirectory() as p:
            for mode in ['PILOT_EXPERIMENT','CONFIRMATORY_EXPERIMENT','HELDOUT_EVALUATION','ENGINEERING_TEST']:
                dest=Path(p)/mode
                r=subprocess.run([sys.executable,str(ROOT/'p7a_run.py'),'--mode',mode,'--config',str(Path(p)/'never-open'),'--output',str(dest)],capture_output=True,text=True)
                self.assertEqual(r.returncode,2);self.assertFalse(dest.exists());self.assertIn('before config/output',r.stderr)
