"""Bounded owner-controlled loopback transactions; external experiments disabled."""
from pathlib import Path
from http.server import HTTPServer,BaseHTTPRequestHandler
from urllib.parse import urlsplit
import http.client,json,re,threading,time
from .contracts import ContractError,sha
from .workloads import atomic
MAX_BODY=65536

class Endpoint:
 def __init__(self,root):
  self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.body_bytes_received=0;self.requests=0;self.events=[]
  owner=self
  class Handler(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def setup(self):super().setup();self.connection.settimeout(2)
   def reply(self,status,data,raw=False):
    body=data if raw else json.dumps(data).encode();self.send_response(status);self.send_header('Content-Length',str(len(body)));self.send_header('Content-Type','application/octet-stream' if raw else 'application/json');self.end_headers();self.wfile.write(body)
   def do_GET(self):
    owner.requests+=1
    if owner.requests>128:return self.reply(429,{'error':'engineering_request_cap'})
    if self.path=='/ping':
     a=time.perf_counter_ns();return self.reply(200,{'server_received_ns':a,'server_reply_prepared_ns':time.perf_counter_ns(),'clock_domain':'server_perf_counter_ns','one_way_latency_ns':None})
    match=re.fullmatch(r'/(receipt|download)/([A-Za-z0-9_-]{1,48})',self.path)
    if not match:return self.reply(404,{'error':'unknown_route'})
    f=owner.root/match[2]
    if not f.exists():return self.reply(404,{'error':'not_found'})
    b=f.read_bytes()
    return self.reply(200,b,True) if match[1]=='download' else self.reply(200,{'application_bytes_persisted':len(b),'sha256':sha(b),'receipt_reopened':True})
   def do_POST(self):
    owner.requests+=1
    if owner.requests>128:return self.reply(429,{'error':'engineering_request_cap'})
    try:
     length=int(self.headers.get('Content-Length','-1'))
     if not 0<=length<=MAX_BODY or owner.body_bytes_received+length>262144:return self.reply(413,{'error':'bounded_original_payload_only'})
     body=self.rfile.read(length)
     if len(body)!=length:return self.reply(400,{'error':'incomplete_body'})
     owner.body_bytes_received+=len(body)
     match=re.fullmatch(r'/upload/([A-Za-z0-9_-]{1,48})/(\d+)',self.path)
     if match:
      f=owner.root/match[1];offset=int(match[2]);old=f.read_bytes() if f.exists() else b''
      if offset==len(old):new=old+body
      elif offset<len(old) and old[offset:offset+len(body)]==body:new=old
      else:return self.reply(409,{'error':'offset_or_duplicate_content_mismatch','persisted_bytes':len(old)})
      if len(new)>MAX_BODY:return self.reply(413,{'error':'upload_cap'})
      atomic(f,new);actual=f.read_bytes();owner.events.append({'event':'application_upload_chunk','offset':offset,'body_bytes_received':len(body),'persisted_total':len(actual),'timestamp_ns':time.perf_counter_ns(),'clock_domain':'server_perf_counter_ns'})
      return self.reply(200,{'application_bytes_persisted':len(actual),'sha256':sha(actual),'receipt_reopened':True})
     if self.path=='/compute':
      x=json.loads(body)
      if set(x)!={'values'} or not isinstance(x['values'],list) or len(x['values'])>128 or any(type(v) is not int or abs(v)>1000000 for v in x['values']):return self.reply(400,{'error':'fixed_sum_input_only'})
      return self.reply(200,{'sum':sum(x['values']),'operation':'bounded_integer_sum','input_sha256':sha(body)})
     return self.reply(404,{'error':'unknown_route'})
    except (ValueError,TimeoutError):return self.reply(400,{'error':'invalid_request'})
  self.server=HTTPServer(('127.0.0.1',0),Handler);self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.05},daemon=True)
 @property
 def url(self):return 'http://127.0.0.1:'+str(self.server.server_address[1])
 def __enter__(self):self.thread.start();return self
 def __exit__(self,*args):self.server.shutdown();self.server.server_close();self.thread.join(timeout=2);assert not self.thread.is_alive()

def request(base,path,body=None):
 u=urlsplit(base)
 # No DNS resolution,redirect,external endpoint or agreement can be triggered.
 if u.scheme!='http' or u.hostname!='127.0.0.1' or u.username or u.password or u.query or u.fragment or u.path not in {'','/'} or not u.port:raise ContractError('Only literal loopback engineering endpoint is enabled;external use requires separate authorized integration')
 if not path.startswith('/') or len(path)>160 or body is not None and len(body)>MAX_BODY:raise ContractError('Bounded transaction')
 start=time.perf_counter_ns();c=http.client.HTTPConnection('127.0.0.1',u.port,timeout=2);data=b'';status=None;failure=None
 try:
  c.request('POST' if body is not None else 'GET',path,body=body);r=c.getresponse();status=r.status;data=r.read(MAX_BODY+1)
  if len(data)>MAX_BODY:raise ContractError('Response cap')
  if status>=400:failure='http_'+str(status)
 except (OSError,TimeoutError,http.client.HTTPException) as e:failure=type(e).__name__
 finally:c.close()
 end=time.perf_counter_ns()
 return {'started_ns':start,'finished_ns':end,'clock_domain':'client_perf_counter_ns','duration_ns':end-start,'http_status':status,'failure':failure,'attempted_body_bytes':len(body) if body is not None else 0,'received_application_bytes':len(data),'response_sha256':sha(data),'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION','observed_transport':'LOOPBACK','packet_loss':None,'packet_loss_reason':'HTTP/TCP transaction errors do not identify lost IP packets','wire_bytes':None},data

def interrupted_upload(base,payload,root):
 if not 2<=len(payload)<=MAX_BODY:raise ContractError('Bounded resumable original payload')
 root=Path(root);mid=len(payload)//2
 first,b=request(base,'/upload/original/0',payload[:mid]);receipt=json.loads(b)
 atomic(root/'progress.json',json.dumps({'offset':receipt.get('application_bytes_persisted',0),'whole_sha256':sha(payload)}).encode())
 # Intentional local application pause,not a public-network outage or physical failure.
 state=json.loads((root/'progress.json').read_text());second,b=request(base,'/upload/original/'+str(state['offset']),payload[state['offset']:]);final=json.loads(b)
 duplicate,db=request(base,'/upload/original/'+str(mid),payload[mid:]);verified,d=request(base,'/download/original')
 correct=first['failure'] is None and second['failure'] is None and verified['failure'] is None and d==payload and final.get('sha256')==sha(payload) and final.get('application_bytes_persisted')==len(payload) and duplicate['failure'] is None and json.loads(db).get('sha256')==sha(payload)
 return {'correct':correct,'progress_preserved':state['offset']==mid,'attempts':[first,second,duplicate,verified],'receipt':final,'recovery_type':'APPLICATION_PAUSE_RESUME_ONLY','actual_wireless_outage':False,'evidence_category':'CONSTRUCTED_ENGINEERING_VERIFICATION'}
