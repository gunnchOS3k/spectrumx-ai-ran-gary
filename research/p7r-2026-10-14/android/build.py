#!/usr/bin/env python3
"""Offline build using already installed SDK/JBR; no Gradle/network/license action."""
from pathlib import Path
import argparse,subprocess,os,json,hashlib,shutil,zipfile
ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--sdk',type=Path,required=True);p.add_argument('--jdk',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sdk=a.sdk;tools=sdk/'build-tools/35.0.0';jar=sdk/'platforms/android-35/android.jar';env={**os.environ,'JAVA_HOME':str(a.jdk),'PATH':str(a.jdk/'bin')+os.pathsep+os.environ.get('PATH','')};log=[]
def run(args):
 r=subprocess.run(list(map(str,args)),env=env,capture_output=True,text=True);log.append({'args':list(map(str,args)),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr});(a.output/'BUILD_LOG.json').write_text(json.dumps(log,indent=2)+'\n');assert r.returncode==0,r.stderr
classes=a.output/'classes';classes.mkdir();dex=a.output/'dex';dex.mkdir();sources=sorted((ROOT/'src').rglob('*.java'))
run([a.jdk/'bin/javac','-source','8','-target','8','-classpath',jar,'-d',classes,*sources])
run([tools/'d8','--min-api','29','--lib',jar,'--output',dex,*sorted(classes.rglob('*.class'))])
base=a.output/'resources.apk';run([tools/'aapt','package','-f','-M',ROOT/'AndroidManifest.xml','-I',jar,'-F',base])
unsigned=a.output/'unsigned.apk'
with zipfile.ZipFile(base) as old,zipfile.ZipFile(unsigned,'w') as z:
 for i in old.infolist():z.writestr(i,old.read(i.filename))
 z.write(dex/'classes.dex','classes.dex')
aligned=a.output/'aligned.apk';run([tools/'zipalign','-f','4',unsigned,aligned]);key=a.output/'local-debug.keystore'
run([a.jdk/'bin/keytool','-genkeypair','-keystore',key,'-storepass','android','-keypass','android','-alias','p7r-local-debug','-keyalg','RSA','-keysize','2048','-validity','365','-dname','CN=P7R LOCAL ENGINEERING DEBUG,O=Not A Research Authorization'])
apk=a.output/'p7r-local-collector-debug.apk';run([tools/'apksigner','sign','--ks',key,'--ks-key-alias','p7r-local-debug','--ks-pass','pass:android','--key-pass','pass:android','--out',apk,aligned]);run([tools/'apksigner','verify','--verbose',apk]);run([tools/'aapt','dump','badging',apk])
h=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
record={'apk_sha256':h(apk),'source_sha256':{str(f.relative_to(ROOT)):h(f) for f in [ROOT/'AndroidManifest.xml',ROOT/'build.py',*sources]},'android_platform_jar_sha256':h(jar),'tool_versions':'Installed SDK35/build-tools35,JBR;see build log','compiled':True,'signature':'Fresh local engineering DEBUG key;not owner identity,publication/pilot authorization or production release','device_installed':False,'device_measured':False,'new_sdk_downloaded_or_terms_accepted':False}
(a.output/'BUILD_VERIFICATION.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
