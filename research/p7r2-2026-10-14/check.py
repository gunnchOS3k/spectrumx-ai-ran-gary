from pathlib import Path
import sys,unittest,json,argparse,subprocess,tempfile,hashlib,platform
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R));p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
s=unittest.defaultTestLoader.discover(str(R/'tests'));n=s.countTestCases()
with (a.output/'TESTS.txt').open('w') as f:r=unittest.TextTestRunner(stream=f,verbosity=2).run(s)
jdk=Path('/Applications/Android Studio.app/Contents/jbr/Contents/Home/bin')
with tempfile.TemporaryDirectory(prefix='p7r2-budget-') as tmp:
 args=[str(jdk/'javac'),'-d',tmp,str(R/'android/src/com/gunnchos/research/p7r/AcceptanceBudget.java'),str(R/'tests/BudgetCheck.java')];b=subprocess.run(args,capture_output=True,text=True);assert b.returncode==0,b.stderr
 b=subprocess.run([str(jdk/'java'),'-cp',tmp,'BudgetCheck'],capture_output=True,text=True);(a.output/'JAVA_BUDGET.txt').write_text(b.stdout+b.stderr);assert b.returncode==0,b.stderr
files={str(f.relative_to(R)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(R.rglob('*')) if f.is_file() and (f.parts[len(R.parts)] in ['android','tests'] or f.parent==R) and '__pycache__' not in f.parts and f.suffix not in ['.pyc']}
x={'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION','python':platform.python_version(),'python_tests_collected':n,'python_tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'skipped':len(r.skipped),'java_budget_result':b.stdout.strip(),'source_sha256':files,'device_or_endpoint_contact':False,'measurement_generated':False,'publication_authorized':False};(a.output/'TEST_RESULT.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({k:v for k,v in x.items() if k!='source_sha256'},indent=2));sys.exit(not r.wasSuccessful())
