# P7-R2 local physical-acceptance preparation

A separate uninstalled candidate,com.gunnchos.research.p7r2 version2/p7r2-0.1,
addresses timing/snapshot/task admission and probe export gaps without replacing
frozen P7-R source or its SHA-256-verified APK. Read the portal p7r2 owner protocol,
commands,endpoint plan and A–M report. No device or endpoint has been measured.

The Activity implements120periodic-snapshot/120s monotonic admission limits,a110s
scheduled early stop,one attempt each of two original tasks,foreground/manual
Stop/export,partial observer-cost fields,and no automatic network transactions.
Android scheduling is not hard realtime. A physical overrun is retained as a
failed acceptance session;do not assert the handler guarantees a wall-time bound.
Callbacks emit transport metadata,not additional full sensor snapshots.

The separately manual instrumentation executes ONE fixed small sum transaction,
only for explicit authorizationRef/exact HTTPS origin and original local window
(start by90s). It does not mint or verify authorization. No redirects,credentials,
query/fragment or path other than /compute.16KiB send/4KiB receive limits,default
TLS verification,no pin bypass. Client timeouts are best effort,not a hard realtime
kill. Response/request byte hashes and original response bytes are exported in a
bounded Base64 instrumentation result with SHA-256,without root/run-as/backup.
An independent host oracle ignores device correctness/expected-result assertions.
The frozen APK is signed with a debug-purpose key but is not a debuggable build;
its private probe capture therefore lacked a documented usable export route.

After explicit future approval only,save raw am instrument stdout;decode and review
using capture_review.py. This script is OFFLINE ONLY and invokes no ADB or network.
It requires the preserved sibling p7r-2026-10-14/src contracts. It rejects malformed,
incomplete,duplicate,wrong-window,over-cap or corrupted captures and retains raw
inputs;known task failures remain failures. Hashes do not prove physical truth.

    python capture_review.py --decode-instrumentation RAW_STDOUT --output NEW
    python capture_review.py --capture LOCAL_JSONL --kind local --output NEW
    python capture_review.py --capture PROBE_JSONL --kind probe --reference-local LOCAL_JSONL --output NEW

Engineering checks use original fixture data only:

    python check.py --output NEW

Python3.14.7 and installed JDK are recorded. Tests use temporary files/local Java,
not sockets,emulators,physical devices,external endpoints or simulation campaigns.
check.py currently locates the already-installed Android Studio JDK on this host;
adapt its explicit path for another machine rather than claiming portable runtime.
No dependencies,SDK terms or services were acquired. android/build.py accepts
--sdk,--jdk,--output NEW and retains complete offline commands/source hashes.
Build artifacts use fresh local engineering signing,not a production owner identity.
Keys/staging are excluded from Git. The final candidate is dist/p7r2-acceptance-candidate.apk.

Retained MIT software notice applies to this derived code;it does not choose an
article/data license,clear patents,authorize installation/deployment or publication.
All physical/scientific/pilot/heldout/publication permissions remain false.
