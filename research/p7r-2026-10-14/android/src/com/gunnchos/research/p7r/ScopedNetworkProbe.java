package com.gunnchos.research.p7r;
import android.net.Network;
import android.os.SystemClock;
import java.net.*;
import java.io.*;
import org.json.JSONObject;
/** Reusable request-scoped adapter,never called automatically by collector Activity. */
public final class ScopedNetworkProbe {
 public static JSONObject transfer(Network network,URL url,String exactApprovedOrigin,boolean independentlyAuthorized,byte[] originalPayload) throws Exception {
  String origin=url.getProtocol()+"://"+url.getAuthority();
  if(!independentlyAuthorized || !origin.equals(exactApprovedOrigin) || !"https".equals(url.getProtocol()) || url.getUserInfo()!=null || originalPayload.length>65536)throw new SecurityException("Exact owner-controlled HTTPS endpoint and separate physical/protocol authorization required");
  long start=SystemClock.elapsedRealtimeNanos();HttpURLConnection c=(HttpURLConnection)(network==null?url.openConnection():network.openConnection(url));c.setInstanceFollowRedirects(false);c.setConnectTimeout(2000);c.setReadTimeout(2000);c.setRequestMethod("POST");c.setDoOutput(true);c.setFixedLengthStreamingMode(originalPayload.length);
  JSONObject r=new JSONObject();int received=0;ByteArrayOutputStream response=new ByteArrayOutputStream();String failure=null;int code=-1;
  try{try(OutputStream out=c.getOutputStream()){out.write(originalPayload);}code=c.getResponseCode();try(InputStream in=code<400?c.getInputStream():c.getErrorStream()){if(in!=null){byte[] b=new byte[2048];int n;while((n=in.read(b))!=-1){if(SystemClock.elapsedRealtimeNanos()-start>5000000000L)throw new java.net.SocketTimeoutException("Overall transfer budget");received+=n;if(received>65536)throw new IOException("Response cap");response.write(b,0,n);}}}if(code>=300)failure="HTTP_"+code;}catch(IOException e){failure=e.getClass().getSimpleName();}finally{c.disconnect();}
  String output="";try{Object value=new JSONObject(response.toString("UTF-8")).get("sum");if(!(value instanceof Integer) && !(value instanceof Long))throw new org.json.JSONException("Integer sum required,no coercion");output=value.toString()+"\n";}catch(Exception ignored){if(failure==null)failure="Missing_expected_sum_output";}
  r.put("canonical_output_utf8",output);r.put("released_ns",start);r.put("finished_ns",SystemClock.elapsedRealtimeNanos());r.put("clock_domain","Android_elapsedRealtimeNanos_boot");r.put("attempted_application_body_bytes",originalPayload.length);r.put("received_application_body_bytes",received);r.put("wire_bytes",JSONObject.NULL);r.put("packet_loss",JSONObject.NULL);r.put("status",code);r.put("failure",failure==null?JSONObject.NULL:failure);r.put("capture_class","UNREVIEWED_DEVICE_CAPTURE");r.put("meaning","Application transaction timing/body counts,not RF/IP loss,oneway latency,carrier RAN control or classified physical evidence");return r;
 }
}
