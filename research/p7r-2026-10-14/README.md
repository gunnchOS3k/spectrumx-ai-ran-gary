# P7-R measurement infrastructure

Intended evidence target: observable real wireless/application performance with
6G research relevance. P7-A synthetic-only pilot is an unapproved historical
alternative. Nothing here runs a prospective field/pilot/testbed campaign.

Python3.14.7 and jsonschema4.26.0 from the existing QA environment were used.
From this source directory, on a host permitting temporary loopback listeners:

```sh
python p7r_check.py --output /tmp/p7r-new-engineering-check
python p7r_run.py --mode PHYSICAL_EXPERIMENT --output /tmp/must-not-exist
```

The first executes only bounded original nonpersonal workloads and loopback
transactions,trace/evaluator/actuator/analysis QA. The second must exit2and
create nothing. External HTTP endpoints are refused before DNS by host client.
Android ScopedNetworkProbe is a dormant,request-scoped HTTPS interface requiring
an independently authorized exact origin;the collector Activity never invokes it.
No private student data,account,IMEI/IMSI/SSID/BSSID/location/root/modem control.

`dist/p7r-local-collector-debug.apk` compiles against installed SDK35 with minSDK29;
its local DEBUG signature was verified. It was NOT installed or run on a device.
ADB device enumeration succeeded but listed zero devices. Complete source/build
logs and prior failed build logs are preserved. Compile success does not validate
Android API behavior,clock accuracy,sensor availability or actual network state.
No SDK/dependencies were downloaded,no new terms accepted. Build manually using
existing SDK/JBR with android/build.py --sdk ... --jdk ... --output NEW_DIRECTORY.
The signing key is an ephemeral local engineering key,not an authorization token
or production signing identity;keys/staging classfiles are not distributed.

Actual host original workloads execute CPython,file read/edit/progress/restore,
and bounded resumable loopback upload/integer-sum endpoints. A byte-identical
MIT-licensed Device OS offline_sync core is vendored with its notice for portable
original-fixture integration checks. This is genuine core-library execution,not
production Capsule/WAIKE execution,cloud receipts,physical transfer,or live data.
The core's own mock:false field is not treated as physical measurement evidence.

Trace schema/rights/source/clock/hash checks do not independently prove sensor
truth. Imported real measurements remain replay,not a new live experiment.
Android captures remain unreviewed until clock/source/protocol/access validation.
All current generated results are CONSTRUCTED_ENGINEERING_VERIFICATION,not
measured5G/NTN/6G evidence. Six acquisition categories are explicit in contracts.py.
Use portal P7R_* reports for evidence gates,standards,testbeds,owners and next steps.

The standalone APK also declares ProbeInstrumentation for one explicitly manual,
scoped HTTPS sum transaction. It requires authorized=true,an actual independently
approved authorizationRef and exact approvedOrigin. It never runs from the launcher.
These arguments are operator attestations,not proof of scientific approval or a
tool-created authorization token. Do not invoke it in this milestone. Its JSONL
output stays UNREVIEWED_DEVICE_CAPTURE until source,clock,path and protocol review.
An owner-controlled TLS endpoint/reverse proxy is required;none was provisioned.
The host endpoint used for QA remains loopback-only. See portal on-device procedure.
