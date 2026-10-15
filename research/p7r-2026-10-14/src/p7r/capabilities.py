from .contracts import ContractError,digest
ACTIONS=('LOCAL_EXECUTION','OWNED_ENDPOINT_OFFLOAD','APP_NETWORK_BIND','APP_SESSION_RESUME','RAN_CONFIG','RADIO_SCHEDULING','PHYSICAL_HANDOVER','NTN_FALLBACK')

def capability(environment,action):
 if action not in ACTIONS or environment not in {'ENGINEERING_LOOPBACK','ANDROID_UNPRIVILEGED','OWNER_WIFI_SERVER','AUTHORIZED_5G_TESTBED','AUTHORIZED_NTN'}:raise ContractError('Capability identity')
 implemented=environment=='ENGINEERING_LOOPBACK' and action in {'LOCAL_EXECUTION','OWNED_ENDPOINT_OFFLOAD','APP_SESSION_RESUME'}
 return {'environment':environment,'action':action,'implemented_and_tested_here':implemented,'physical_execution_authorized':False,'requires_access_and_protocol':not implemented,'carrier_radio_control':False,'rationale':'Application-level action only;external/testbed/NTN actuation requires exact authorized interface and independent capability evidence'}

def matched_contract(a,b):
 keys={'observations','eligible_actions','resource_limits','compute_limits','switch_costs','endpoint_scope','missing_observation_rule'}
 if set(a)!=keys or set(b)!=keys or digest(a)!=digest(b):raise ContractError('Unequal information/action/resource/compute/cost opportunity')
 return True

def research_guard(mode):raise ContractError('P7-R field/pilot/heldout/publication execution disabled;only constructed engineering checks allowed')
