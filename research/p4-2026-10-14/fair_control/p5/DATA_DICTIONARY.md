# P5 engineering result dictionary

ENGINEERING_RUN.json: payload (canonical deterministic engineering data); payload_sha256; timing_provenance (retained elapsed seconds); environment (Python/platform/pinned versions,not fresh install); manifest (every input hash/source-tree hash).

payload.mode always ENGINEERING_TEST; fixture_id identifies one source-pinned nonrandom fixture. source_sha records exact Git/content revision; configuration_sha256 identifies original full config; controller/model/environment_revision identify source module byte hashes. seed_manifest has CONSTRUCTED_NO_RANDOMNESS,seed_ids:[],heldout_paths:[]. No future seed streams are provisioned.

primary_outcome: scheduled count of all TaskSpecs,successes verified successful outputs,fraction=successes/scheduled, plus explicit endpoint schema name. secondary_outcome: energy_used_j (dimensioned simulation), queue_length,reasons for failed tasks,invalid action count. No p-value,confidence interval or population effect is created by this fixture.

decisions: action(network,placement,allocations,fidelity),status,actual model evaluations,consumed fields,observation/catalog hashes,predicted completion score or null,error or null. Timing retained separately in matching index order,validated finite/nonnegative. These are engineering records,not favorable policy traces selected for a paper.

checkpoint: version,specs hash,Model,now_s,initial/remaining energy,previous executed Action, per-task state/progress/delivery/latency/path/place/history/checkpoint/failure,content bytes and generated timestamps,verified successful observed outcomes,all attempted finished outputs including rejected bytes,raw execution/maintenance trace. Synthetic-only inputs contain no private data. Content_age_s is age of oldest required dependency at observed finish;quality normalized;durable denotes retained synthetic state.

SOURCE_PROVENANCE.json: exact file map plus canonical tree hash; SOURCE_REVISION.json: clean-copy implementation-content anchor distinct from future package/governance commit. RUN_STATE.json: engineering COMPLETED state,payload hash,false approval fields; never a research-run token. Error before guard/config acceptance does not create a run-state directory. Existing future-result.schema.json remains unpopulated; old statistical summaries/protocols are not new research evidence.
