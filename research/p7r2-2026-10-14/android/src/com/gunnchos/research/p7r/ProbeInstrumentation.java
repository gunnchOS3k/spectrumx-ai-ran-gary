package com.gunnchos.research.p7r2;
import android.app.*;
import android.content.*;
import android.net.*;
import android.os.*;
import org.json.*;
import java.io.*;
import java.net.URL;
import java.security.MessageDigest;
import java.util.Locale;
/** Explicit manual adb instrumentation only;never started by launcher/collector. */
public final class ProbeInstrumentation extends Instrumentation {
 private Bundle args;
 private File trace;
 private int seq=0;
 private void exportResult(Bundle result)throws Exception{if(trace==null)return;try(FileInputStream in=new FileInputStream(trace);ByteArrayOutputStream out=new ByteArrayOutputStream()){byte[] buf=new byte[2048];int n;while((n=in.read(buf))!=-1){if(out.size()+n>65536)throw new IOException("Export cap");out.write(buf,0,n);}String raw=out.toString("UTF-8");result.putString("p7r_capture_base64",android.util.Base64.encodeToString(raw.getBytes("UTF-8"),android.util.Base64.NO_WRAP));result.putString("p7r_capture_sha256",hash(raw));}}
 @Override public void onCreate(Bundle a){super.onCreate(a);args=a;start();}
 private String hash(String s)throws Exception{byte[] b=MessageDigest.getInstance("SHA-256").digest(s.getBytes("UTF-8"));StringBuilder x=new StringBuilder();for(byte v:b)x.append(String.format(Locale.ROOT,"%02x",v&255));return x.toString();}
 private void emit(String kind,JSONObject data)throws Exception{long uptime=SystemClock.uptimeMillis();JSONObject r=new JSONObject();r.put("schema","p7r.android.event.v1");r.put("seq",seq++);r.put("uptime_ms",uptime);r.put("elapsed_ns",SystemClock.elapsedRealtimeNanos());r.put("kind",kind);r.put("data",data);r.put("capture_class","UNREVIEWED_DEVICE_CAPTURE");try(FileOutputStream s=new FileOutputStream(trace,true)){s.write((r.toString()+"\n").getBytes("UTF-8"));s.getFD().sync();}}
 @Override public void onStart(){
  Bundle result=new Bundle();
  try{
   String endpoint=args.getString("endpoint","");String origin=args.getString("approvedOrigin","");String ref=args.getString("authorizationRef","");
   if(!"true".equals(args.getString("authorized","false"))||ref.length()==0||origin.length()==0)throw new SecurityException("Separate owner-approved protocol/exact origin/access reference required;no approval inferred from app");
   URL url=new URL(endpoint);if(!"https".equals(url.getProtocol())||!origin.equals(url.getProtocol()+"://"+url.getAuthority()))throw new SecurityException("Exact authorized HTTPS origin required");
   long approvedWindowStart=Long.parseLong(args.getString("windowStartElapsedNs","-1"));long now=SystemClock.elapsedRealtimeNanos();if(approvedWindowStart<0||now<approvedWindowStart||now-approvedWindowStart>90000000000L)throw new SecurityException("Original local capture window required;start probe by90seconds,no fresh window/retry");
   Context c=getTargetContext();trace=new File(c.getFilesDir(),"p7r-probe-"+SystemClock.elapsedRealtimeNanos()+".jsonl");JSONObject start=new JSONObject();start.put("collector_version","p7r2-0.1");start.put("capture_scope","One original bounded sum transaction;operator attestation,not verified scientific approval");start.put("source_protocol_approved_by_tool",false);start.put("window_start_elapsed_ns",approvedWindowStart);emit("session_start",start);
   ConnectivityManager cm=(ConnectivityManager)c.getSystemService(Context.CONNECTIVITY_SERVICE);Network n=cm.getActiveNetwork();NetworkCapabilities cap=cm.getNetworkCapabilities(n);JSONObject obs=new JSONObject();obs.put("os_cellular_transport",cap!=null&&cap.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR));obs.put("os_wifi_transport",cap!=null&&cap.hasTransport(NetworkCapabilities.TRANSPORT_WIFI));obs.put("meaning","Default route capabilities,not carrier control or satellite path proof");emit("snapshot",obs);
   JSONArray values=new JSONArray();for(int i=1;i<=10;i++)values.put(i);JSONObject input=new JSONObject();input.put("values",values);
   JSONObject transfer=ScopedNetworkProbe.transfer(n,url,origin,true,input.toString().getBytes("UTF-8"));String output=transfer.getString("canonical_output_utf8");transfer.put("task_id","original-endpoint-sum");transfer.put("expected_sha256",hash("55\n"));transfer.put("observed_output_sha256",hash(output));transfer.put("durable_reopened",false);transfer.put("correct","55\n".equals(output)&&transfer.isNull("failure"));transfer.put("scope","Original benchmark computation,not production application or calibrated network measurement");transfer.remove("canonical_output_utf8");transfer.remove("capture_class");emit("task_result",transfer);emit("session_stop",new JSONObject());
   result.putString("private_capture_file",trace.getName());result.putString("classification","UNREVIEWED_DEVICE_CAPTURE");exportResult(result);finish(Activity.RESULT_OK,result);
  }catch(Exception e){result.putString("error",e.getClass().getSimpleName()+": "+e.getMessage());try{if(trace!=null){JSONObject err=new JSONObject();err.put("reason",e.getClass().getSimpleName());emit("collector_error",err);emit("session_stop",new JSONObject());}}catch(Exception ignored){}try{exportResult(result);}catch(Exception ignored){}finish(Activity.RESULT_CANCELED,result);}
 }
}
