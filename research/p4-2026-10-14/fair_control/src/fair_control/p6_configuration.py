"""Complete draft configuration; validation never grants execution permission."""
from pathlib import Path
import json,hashlib
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError
from .core import ContractError,canonical,digest
from .controllers import Policy,Hardware,FIELDS
from .dynamics import Model
from .p6_application import catalog
from .p6_scenarios import Gate,Link,CausalEstimator
ROOT=Path(__file__).resolve().parents[2]
AUTH={'scientific_protocol_approved':False,'pilot_execution_authorized':False,'heldout_execution_authorized':False,'human_scientific_approval':False,'publication_authorized':False}
MODES=['ENGINEERING_TEST','PILOT_EXPERIMENT','CONFIRMATORY_EXPERIMENT','HELDOUT_EVALUATION']
STATUS=['IMPLEMENTED_AND_VERIFIED','PROVISIONAL_ENGINEERING_DEFAULT','PROPOSED_SCIENTIFIC_PARAMETER','CALIBRATION_REQUIRED','OWNER_DECISION_REQUIRED','SCIENTIFICALLY_UNSUPPORTED']


def parameter(value,units,meaning, *,status='PROPOSED_SCIENTIFIC_PARAMETER',source='P4/P5 locally authored proposal',justification='Constructed synthetic stress assumption; no physical calibration or practical-effect evidence'):
    return {'value':value,'units':units,'meaning':meaning,'classification':status,'source':source,'scientific_justification':justification,'uncertainty':'Not empirically estimated','dependencies':['owner/domain scientific review'],'validation_required':'Dimensional/domain checks and prospective applicability review','owner_approved':False}

def source_identities(source_sha):
    names=['core.py','tasks.py','dynamics.py','controllers.py','fairness.py','p6_application.py','p6_scenarios.py','p6_process.py','p6_calibration.py','p6_statistics.py','p6_configuration.py']
    return {'source_sha':source_sha,'source_sha_role':'Implementation-content anchor; configuration bytes are separately hashed','modules':{n:hashlib.sha256((ROOT/'src/fair_control'/n).read_bytes()).hexdigest() for n in names},'task_catalog_sha256':digest(catalog())}

def candidate(source_sha,protocol_sha256):
    p={}
    for name,value,unit,meaning in [
        ('control_interval',1.,'s','Decision/frame interval; only1s current predictor supported'),('cpu_capacity',10.,'Mcycles/s','Global allocated computation ceiling'),('local_cpu',5.,'Mcycles/s','Local placement ceiling'),('memory',100.,'MB','Allocated task memory ceiling'),('bandwidth',10.,'Mbps','Terminal aggregate data ceiling'),('power',2.,'W','Full-slot allocated synthetic power ceiling'),('initial_energy',100.,'J','Finite assumed initial battery; depletion retained'),('network_switch',.05,'s','Additive current path interruption'),('network_switch_energy',.01,'J','Assumed path switch energy'),('placement_switch',.1,'s','Additive migration interruption'),('placement_switch_energy',.02,'J','Assumed placement energy'),('hysteresis',.05,'predicted completed tasks','Retention margin on one-step predicted task count'),('max_model_calls',512,'calls/decision','Shared maximum predictor opportunity,actual work reported'),('decision_deadline',.1,'s','Input/IPC/controller decision window excluding separately capped startup'),('startup_cap',1.,'s','Separate common child startup allowance'),('margin',.02,'completion-fraction difference','Practical improvement null boundary'),('alternative',.04,'completion-fraction difference','Hypothetical planning alternative'),('hypothetical_sd',.15,'paired fraction SD','Planning assumption,not pilot observation'),('half_width',.01,'completion-fraction difference','Proposed95% precision target'),('target_power',.9,'probability','Conditional planning power target'),('pilot_pairs',40,'independent pairs','Feasibility/variance purpose only'),('heldout_min',30,'independent pairs','Future minimum after planning'),('heldout_cap',2500,'independent pairs','Future ceiling before outcome access'),('episodes_per_unit',24,'episodes','Proposed cluster size per seed mean'),('steps_per_episode',64,'steps','Proposed episode horizon'),('max_tasks_episode',128,'tasks','Predeclared exogenous workload cap; excess generation invalid before decisions'),('cpu_hours_cap',12.,'CPU-hours','Proposed whole future study pilot+tuning+evaluation cap'),('observation_delay',1.,'s','Authored delayed-history example'),('capacity_relative_error',-.2,'fraction','Authored observed capacity bias'),('latency_bias',.02,'s','Authored estimate delay bias'),('poisson_rate',.8,'tasks/s','Authored additional arrival rate'),('background_load',.2,'capacity fraction','Authored contention subtraction')]:
        p[name]=parameter(value,unit,meaning)
    p['compute_service_treatment']=parameter('recorded_not_debited','categorical','Host compute time separately measured; not converted into simulated service delay',status='OWNER_DECISION_REQUIRED',justification='Explicit discrete-model convention. Deployment/control-latency superiority cannot be inferred; owner may require a common reserved window or causal compute debit before pilot.')
    gate_rows=[{'name':'common_power','fail_rate_s':.005,'restore_rate_s':.2},{'name':'TN','fail_rate_s':.05,'restore_rate_s':.25},{'name':'Wi-Fi','fail_rate_s':.02,'restore_rate_s':.5},{'name':'NTN_gateway','fail_rate_s':.02,'restore_rate_s':.5}]
    links=[{'name':'TN','capacity_mbps':10.,'latency_s':.02,'dependencies':['common_power','TN']},{'name':'Wi-Fi','capacity_mbps':5.,'latency_s':.04,'dependencies':['common_power','Wi-Fi']},{'name':'NTN','capacity_mbps':3.,'latency_s':.2,'dependencies':['common_power','NTN_gateway']}]
    return {'schema_version':'p6-freeze-candidate-0.1','experiment_id':'paper-ii-prospective-task-control-p6-draft','execution_mode':'PILOT_EXPERIMENT','review_state':'DRAFT_OWNER_APPROVAL_REQUIRED','protocol_revision':{'path':'research/p4-2026-10-14/PROSPECTIVE_PROTOCOL.md','sha256':protocol_sha256,'independently_registered':False},'source':source_identities(source_sha),'authorization':dict(AUTH),'configuration_status':{'configuration_complete':True,'configuration_technically_validated':False,'scientific_design_approved':False,'protocol_frozen_for_pilot':False,'pilot_execution_authorized':False},'parameters':p,'calibration':{'model_mode':'SYNTHETIC_ONLY','physical_calibration_available':False,'measurement_records':[],'standards_are_measurements':False},'task_catalog':catalog(),'scenario':{'version':'p6-causal-scenario-0.1','parameter_status':'SYNTHETIC_ASSUMPTION_OWNER_REVIEW_REQUIRED','gates':gate_rows,'links':links,'arrival':'poisson_supplied_uniform','burst_model':'constructed_count_input','initial_task':'One exogenous eligible task at0 to avoid empty unit denominators; extra arrivals independent of policy','background_load_fraction':.2,'strata':['independent_components','shared_power','delayed_biased_observation','capacity_contention'],'mixture_weights':[.25,.25,.25,.25],'heldout_regime_values':None,'units':'rates/s;latency/holding time s;capacity Mbps;data Mbit'},'observations':{'fields':list(FIELDS),'estimator':'causal-persistence-0.1','delay_s':1.,'capacity_relative_error':-.2,'latency_bias_s':.02,'availability_flip':False,'forecast_horizon':3,'future_truth_access':False},'hardware':{'cpu_mcycles_s':10.,'memory_mb':100.,'bandwidth_mbps':10.,'power_w':2.,'placement_cpu':[['local',5.],['edge',10.],['cloud',10.]]},'model':{'network_stall_s':.05,'network_energy_j':.01,'placement_stall_s':.1,'placement_energy_j':.02,'degraded_quality':.5},'policies':[{'id':'beam_mpc_candidate','family':'mpc','horizon':3,'beam_width':4,'hysteresis':.05,'max_evaluations':512,'deadline_s':.1,'observations':list(FIELDS)},{'id':'matched_hysteresis_candidate','family':'hysteresis','horizon':3,'beam_width':4,'hysteresis':.05,'max_evaluations':512,'deadline_s':.1,'observations':list(FIELDS)}],'compute':{'wrapper':'p6_ready_handshake','startup_allowance_s':1.,'decision_allowance_s':.1,'actual_calls_and_all_time_components_reported':True,'service_delay_treatment':'recorded_not_debited','real_time_certified':False},'outcomes':{'primary':'all_exogenous_eligible_tasks_correct_deadline_quality_fresh_durable_fraction','weighting':'equal_independent_seed_unit_means','secondary':['p95_latency_among_successes_with_fraction','energy_per_attempt','queue_and_deadline_failures','invalid_action_count','switch_stall_and_count','actual_controller_compute_and_startup'],'empty_unit':'generator_invalid_before_policy_execution','controller_failure':'defer;tasks_remain_denominator','task_failure':'zero_success_in_fixed_denominator','harness_failure':'retain pair/error;stop for deviation review,no selective deletion'},'analysis':{'primary_contrast':'MPC_minus_hysteresis','null':'mean_difference<=0.02','practical_margin':.02,'planning_alternative':.04,'confidence_interval':'two_sided_95_percent_paired_Student_t_on_independent_unit_means','alpha':.05,'power_target':.9,'half_width_target':.01,'pilot_pairs_proposal':40,'hypothetical_sd':.15,'sample_cap':2500,'secondary_confirmatory':'Holm_predeclared_family','pilot_reused_confirmatory':False,'optional_stopping':False,'tuning_split_separate':True,'heldout_validation_separate':True},'randomization':{'method':'Future disjoint SHA256-derived streams after approved study freeze','seed_manifest':None,'pilot_seed_ids':None,'tuning_seed_ids':None,'confirmatory_seed_ids':None,'heldout_seed_ids':None},'holdout_access':{'authorized':False,'outcome_paths':[],'reserved_regimes_read':False,'pilot_tuning_confirmatory_validation_disjoint':True},'stop_conditions':['invalid_configuration_or_source_identity','uncleared_external_rights','resource_or_causal_invariant','sample_precision_or_runtime_cap_infeasible_before_holdout','unexpected_harness_failure_or_protocol_deviation','no_execution_authorization'],'pilot_bounds':{'max_pairs':40,'episodes_per_pair':24,'steps_per_episode':64,'max_tasks_per_episode':128,'max_simulation_s_per_policy_unit':1536.,'max_cpu_hours_total':12.,'output_path_proposal':'research_runs/UNEXECUTED/pilot/<frozen-config-sha>/','data_classification':'ORIGINAL_SYNTHETIC_NONPERSONAL','executed':False},'rights':{'external_dataset_imports':[],'WAIKE_rights_cleared':False,'original_fixture_local_engineering':True,'article_license_approved':False,'patent_clearance':False,'additional_authors_funding_conflicts':'UNKNOWN_OWNER_REVIEW','external_application_execution_permitted':False},'owner_decisions':['scope_and_task_semantics','physical_or_synthetic_only_claims','fidelity_tradeoff','forecast_arrival_outage_model','compute_service_delay_treatment','effect_and_sampling','rights_and_patents','scientific_protocol_freeze','separate_pilot_authorization']}


def validate_candidate(c, *,check_sources=True):
    try:Draft202012Validator(json.loads((ROOT/'schemas/p6-configuration.schema.json').read_text())).validate(c)
    except ValidationError as ex:raise ContractError('P6 schema: '+ex.message) from ex
    canonical(c)
    if c['authorization']!={k:False for k in ('scientific_protocol_approved','pilot_execution_authorized','heldout_execution_authorized','human_scientific_approval','publication_authorized')} or any(v is True for k,v in c['configuration_status'].items() if k in {'scientific_design_approved','protocol_frozen_for_pilot','pilot_execution_authorized'}):raise ContractError('Unapproved design cannot authorize execution')
    if c['calibration']['physical_calibration_available'] or c['calibration']['measurement_records']:raise ContractError('P6 has no imported measurements')
    if c['source']['task_catalog_sha256']!=digest(c['task_catalog']):raise ContractError('Task catalog revision')
    if c['parameters']['control_interval']['value']!=1.:raise ContractError('Current predictor requires1s;other intervals unsupported until revised/tested')
    if c['parameters']['alternative']['value']<=c['parameters']['margin']['value']:raise ContractError('Alternative must exceed practical null margin')
    if c['analysis']['practical_margin']!=c['parameters']['margin']['value'] or c['analysis']['planning_alternative']!=c['parameters']['alternative']['value']:raise ContractError('Hypothesis/parameter mismatch')
    if c['analysis']['null']!=f"mean_difference<={c['analysis']['practical_margin']}":raise ContractError('Hypothesis text mismatch')
    if any(p['owner_approved'] for p in c['parameters'].values()):raise ContractError('No silent parameter approval')
    if c['randomization']['seed_manifest'] is not None or any(v is not None for k,v in c['randomization'].items() if k.endswith('_seed_ids')):raise ContractError('No future seeds in P6')
    if c['holdout_access']['authorized'] or c['holdout_access']['outcome_paths'] or c['holdout_access']['reserved_regimes_read'] or c['scenario']['heldout_regime_values'] is not None:raise ContractError('Reserved holdout prohibited')
    h=Hardware(**{**c['hardware'],'placement_cpu':tuple(tuple(x) for x in c['hardware']['placement_cpu'])});Model(**c['model'])
    ps=[Policy(**{k:tuple(v) if k=='observations' else v for k,v in p.items() if k!='id'}) for p in c['policies']]
    if [p.family for p in ps]!=['mpc','hysteresis'] or any(p.horizon!=3 or p.beam_width!=4 for p in ps):raise ContractError('Primary H3/width4 comparison')
    if ps[0].observations!=ps[1].observations or ps[0].max_evaluations!=ps[1].max_evaluations or ps[0].deadline_s!=ps[1].deadline_s:raise ContractError('Unequal primary opportunity')
    if c['compute']['decision_allowance_s']!=ps[0].deadline_s or c['compute']['service_delay_treatment']!=c['parameters']['compute_service_treatment']['value']:raise ContractError('Compute accounting conflict')
    for row in c['scenario']['gates']:Gate(**row)
    gates={x['name'] for x in c['scenario']['gates']}
    for row in c['scenario']['links']:
        l=Link(**{**row,'dependencies':tuple(row['dependencies'])})
        if not set(l.dependencies)<=gates:raise ContractError('Scenario dependency not defined')
    weights=c['scenario']['mixture_weights']
    if len(weights)!=len(c['scenario']['strata']) or not abs(sum(weights)-1)<1e-12:raise ContractError('Scenario mixture')
    CausalEstimator(c['observations']['delay_s'],c['observations']['capacity_relative_error'],c['observations']['latency_bias_s'],c['observations']['availability_flip'])
    # Reject conflicting duplicate declarations, rather than silently preferring one.
    bindings={
      'cpu_capacity':c['hardware']['cpu_mcycles_s'],'local_cpu':dict(c['hardware']['placement_cpu'])['local'],
      'memory':c['hardware']['memory_mb'],'bandwidth':c['hardware']['bandwidth_mbps'],'power':c['hardware']['power_w'],
      'network_switch':c['model']['network_stall_s'],'network_switch_energy':c['model']['network_energy_j'],
      'placement_switch':c['model']['placement_stall_s'],'placement_switch_energy':c['model']['placement_energy_j'],
      'hysteresis':ps[1].hysteresis,'max_model_calls':ps[0].max_evaluations,'decision_deadline':ps[0].deadline_s,
      'startup_cap':c['compute']['startup_allowance_s'],'margin':c['analysis']['practical_margin'],
      'alternative':c['analysis']['planning_alternative'],'hypothetical_sd':c['analysis']['hypothetical_sd'],
      'half_width':c['analysis']['half_width_target'],'target_power':c['analysis']['power_target'],
      'pilot_pairs':c['analysis']['pilot_pairs_proposal'],'heldout_cap':c['analysis']['sample_cap'],
      'episodes_per_unit':c['pilot_bounds']['episodes_per_pair'],'steps_per_episode':c['pilot_bounds']['steps_per_episode'],
      'max_tasks_episode':c['pilot_bounds']['max_tasks_per_episode'],'cpu_hours_cap':c['pilot_bounds']['max_cpu_hours_total'],
      'observation_delay':c['observations']['delay_s'],'capacity_relative_error':c['observations']['capacity_relative_error'],
      'latency_bias':c['observations']['latency_bias_s'],'background_load':c['scenario']['background_load_fraction']}
    if any(c['parameters'][k]['value']!=v for k,v in bindings.items()):raise ContractError('Parameter/declaration inconsistency')
    if c['pilot_bounds']['max_pairs']!=c['analysis']['pilot_pairs_proposal'] or c['pilot_bounds']['max_simulation_s_per_policy_unit']!=c['parameters']['control_interval']['value']*c['pilot_bounds']['episodes_per_pair']*c['pilot_bounds']['steps_per_episode']:raise ContractError('Pilot simulation bounds')
    if not 0<c['analysis']['practical_margin']<c['analysis']['planning_alternative']<=1 or not 0<c['analysis']['half_width_target']<1 or not .5<c['analysis']['power_target']<1:raise ContractError('Scientific fraction/power domain')
    if not 30<=c['analysis']['sample_cap']<=2500 or not 1<=c['analysis']['pilot_pairs_proposal']<=40:raise ContractError('Bounded proposal sizes')
    if c['compute']['wrapper']!='p6_ready_handshake' or c['compute']['real_time_certified'] or c['observations']['future_truth_access']:raise ContractError('Unsupported process/truth claim')
    if c['outcomes']['primary']!='all_exogenous_eligible_tasks_correct_deadline_quality_fresh_durable_fraction' or c['outcomes']['weighting']!='equal_independent_seed_unit_means':raise ContractError('Endpoint incompatibility')
    if c['analysis']['pilot_reused_confirmatory'] or c['analysis']['optional_stopping'] or not c['analysis']['tuning_split_separate'] or not c['analysis']['heldout_validation_separate']:raise ContractError('Stage independence')
    if c['rights']['external_dataset_imports'] or c['rights']['WAIKE_rights_cleared'] or c['rights']['external_application_execution_permitted'] or c['rights']['article_license_approved'] or c['rights']['patent_clearance']:raise ContractError('No implicit rights clearance')
    if c['pilot_bounds']['executed']:raise ContractError('Pilot must remain unexecuted')
    if check_sources:
        if source_identities(c['source']['source_sha'])!=c['source']:raise ContractError('Changed source/config provenance')
    return {'configuration_complete':True,'configuration_technically_validated':True,'scientific_design_approved':False,'protocol_frozen_for_pilot':False,'pilot_execution_authorized':False,'candidate_sha256':digest(c)}

def verify_seal(c,seal):
    validate_candidate(c)
    if seal!={'candidate_sha256':digest(c),'source':c['source'],'authorization':AUTH}:raise ContractError('Freeze seal altered scientific config/provenance')
    return True

def execution_guard(mode,config,output=None):
    # Source-disabled research guard comes BEFORE schema/data/output access.
    if mode!='ENGINEERING_TEST':raise ContractError('P6 research execution disabled; separate governance authorization required')
    raise ContractError('Freeze candidate is review-only; only p6_check.py constructed tests may execute')
