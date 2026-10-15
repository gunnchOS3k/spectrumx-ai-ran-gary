package com.gunnchos.research.p7r2;

import android.Manifest;
import android.app.Activity;
import android.content.*;
import android.content.pm.PackageManager;
import android.net.*;
import android.os.*;
import android.telephony.*;
import android.widget.*;
import org.json.*;
import java.io.*;
import java.security.MessageDigest;
import java.util.*;

/** Standalone, manual, local-only instrumentation. No production app or modem control. */
public final class CollectorActivity extends Activity {
  private final Handler handler = new Handler(Looper.getMainLooper());
  private ConnectivityManager connectivity;
  private ConnectivityManager.NetworkCallback callback;
  private File trace;
  private boolean recording=false;
  private int seq=0;
  private AcceptanceBudget budget;
  private long previousAppendNs=0,previousAppendCpuMs=0;
  private final Runnable hardStop=()->stopCapture();
  private TextView status;
  private final Runnable tick=new Runnable(){public void run(){if(recording){long now=SystemClock.elapsedRealtimeNanos();if(!budget.snapshot(now)){stopCapture();return;}long before=SystemClock.elapsedRealtimeNanos(),cpu=android.os.Process.getElapsedCpuTime();JSONObject data=snapshot();put(data,"snapshot_collection_ns",SystemClock.elapsedRealtimeNanos()-before);put(data,"snapshot_collection_cpu_ms",android.os.Process.getElapsedCpuTime()-cpu);emit("snapshot",data);if(recording)handler.postDelayed(this,1000);}}};
  @Override public void onCreate(Bundle b){
    super.onCreate(b);connectivity=(ConnectivityManager)getSystemService(CONNECTIVITY_SERVICE);
    LinearLayout layout=new LinearLayout(this);layout.setOrientation(LinearLayout.VERTICAL);layout.setPadding(24,24,24,24);
    status=new TextView(this);status.setText("P7-R unreviewed local instrumentation. No campaign is approved. Start only after separate owner authorization. No private content, location, SSID, SIM or device IDs are collected. No network probe runs automatically.");layout.addView(status);
    button(layout,"Start bounded local capture",()->startCapture());
    button(layout,"Optional phone telemetry permission",()->requestPermissions(new String[]{Manifest.permission.READ_PHONE_STATE},10));
    button(layout,"Original local document task",()->localTask(false));
    button(layout,"Original bounded computation task",()->localTask(true));
    button(layout,"Stop",()->stopCapture());
    button(layout,"Export locally",()->{stopCapture();if(trace!=null){Intent i=new Intent(Intent.ACTION_CREATE_DOCUMENT);i.setType("application/x-ndjson");i.putExtra(Intent.EXTRA_TITLE,"p7r-unreviewed-local.jsonl");startActivityForResult(i,11);}});
    setContentView(layout);
  }
  private void button(LinearLayout l,String t,Runnable r){Button b=new Button(this);b.setText(t);b.setOnClickListener(v->r.run());l.addView(b);}
  private void startCapture(){
    if(recording)return;
    // Do not overwrite an earlier capture; app-private original benchmark files only.
    trace=new File(getFilesDir(),"p7r-"+SystemClock.elapsedRealtimeNanos()+".jsonl");seq=0;recording=true;budget=new AcceptanceBudget(SystemClock.elapsedRealtimeNanos());previousAppendNs=0;previousAppendCpuMs=0;handler.postDelayed(hardStop,110000);
    JSONObject id=new JSONObject();put(id,"manufacturer",Build.MANUFACTURER);put(id,"model",Build.MODEL);put(id,"sdk",Build.VERSION.SDK_INT);put(id,"build_incremental",Build.VERSION.INCREMENTAL);put(id,"protocol_approved",false);put(id,"collector_version","p7r2-0.1");put(id,"window_start_elapsed_ns",budget.startNs);put(id,"maximum_duration_ns",AcceptanceBudget.MAX_NS);put(id,"maximum_periodic_snapshots",120);emit("session_start",id);
    callback=new ConnectivityManager.NetworkCallback(){
      @Override public void onAvailable(Network n){emit("network_available",network(connectivity.getNetworkCapabilities(n)));}
      @Override public void onLost(Network n){JSONObject o=new JSONObject();put(o,"meaning","Previously default network lost;not necessarily total service loss or packet-loss count");emit("network_lost",o);}
      @Override public void onCapabilitiesChanged(Network n,NetworkCapabilities c){emit("network_capabilities",network(c));}
    };
    try{connectivity.registerDefaultNetworkCallback(callback,handler);}catch(RuntimeException e){JSONObject o=new JSONObject();put(o,"reason",e.getClass().getSimpleName());emit("collector_error",o);}
    handler.post(tick);status.setText("Recording unreviewed local data, maximum120snapshots/120seconds(and300total events). Foreground only. No live testbed/NTN/6G validation.");
  }
  private void stopCapture(){
    if(!recording)return;emit("session_stop",new JSONObject());recording=false;handler.removeCallbacks(tick);handler.removeCallbacks(hardStop);
    if(callback!=null){try{connectivity.unregisterNetworkCallback(callback);}catch(RuntimeException ignored){}callback=null;}
    status.setText("Stopped. Captures remain app-private until manual export. Physical validity and clock calibration remain unverified.");
  }
  @Override protected void onStop(){super.onStop();stopCapture();}
  private static void put(JSONObject o,String k,Object v){try{o.put(k,v==null?JSONObject.NULL:v);}catch(JSONException e){throw new IllegalStateException(e);}}
  private synchronized void emit(String kind,JSONObject data){
    if(!recording)return;
    if(!"session_stop".equals(kind)&&!budget.active(SystemClock.elapsedRealtimeNanos())){stopCapture();return;}
    put(data,"previous_trace_append_ns",previousAppendNs);put(data,"previous_trace_append_cpu_ms",previousAppendCpuMs);
    if(seq>=299 && !"session_stop".equals(kind)){stopCapture();return;}
    if(seq>=300)return;
    JSONObject o=new JSONObject();put(o,"schema","p7r.android.event.v1");put(o,"seq",seq++);long uptime=SystemClock.uptimeMillis();long elapsed=SystemClock.elapsedRealtimeNanos();put(o,"elapsed_ns",elapsed);put(o,"uptime_ms",uptime);put(o,"kind",kind);put(o,"data",data);put(o,"capture_class","UNREVIEWED_DEVICE_CAPTURE");
    long appendStart=SystemClock.elapsedRealtimeNanos(),appendCpu=android.os.Process.getElapsedCpuTime();
    try(FileOutputStream f=new FileOutputStream(trace,true)){f.write((o.toString()+"\n").getBytes("UTF-8"));f.flush();previousAppendNs=SystemClock.elapsedRealtimeNanos()-appendStart;previousAppendCpuMs=android.os.Process.getElapsedCpuTime()-appendCpu;}
    catch(IOException e){recording=false;handler.removeCallbacks(tick);handler.removeCallbacks(hardStop);if(callback!=null){try{connectivity.unregisterNetworkCallback(callback);}catch(RuntimeException ignored){}callback=null;}status.setText("Capture write failed;session stopped: "+e.getClass().getSimpleName());}
  }
  private JSONObject network(NetworkCapabilities c){
    JSONObject o=new JSONObject();JSONArray t=new JSONArray();
    if(c!=null){if(c.hasTransport(NetworkCapabilities.TRANSPORT_WIFI))t.put("WIFI");if(c.hasTransport(NetworkCapabilities.TRANSPORT_CELLULAR))t.put("CELLULAR");if(c.hasTransport(NetworkCapabilities.TRANSPORT_ETHERNET))t.put("ETHERNET");if(c.hasTransport(NetworkCapabilities.TRANSPORT_VPN))t.put("VPN");put(o,"internet_capability",c.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET));put(o,"validated_capability",c.hasCapability(NetworkCapabilities.NET_CAPABILITY_VALIDATED));put(o,"not_metered_capability",c.hasCapability(NetworkCapabilities.NET_CAPABILITY_NOT_METERED));}
    put(o,"transports",t);put(o,"active_network_available",c!=null);put(o,"interpretation","OS default capabilities,not RF scheduler,handover proof or reachable measurement endpoint");return o;
  }
  private JSONObject unavailable(String unit,String why){JSONObject o=new JSONObject();put(o,"value",null);put(o,"unit",unit);put(o,"status","UNAVAILABLE");put(o,"reason",why);return o;}
  private JSONObject metric(long v,String unit){if(v==Long.MIN_VALUE || v==Integer.MIN_VALUE || v==Integer.MAX_VALUE)return unavailable(unit,"API sentinel/unsupported");JSONObject o=new JSONObject();put(o,"value",v);put(o,"unit",unit);put(o,"status","AVAILABLE");put(o,"reason",null);return o;}
  private JSONObject snapshot(){
    JSONObject o=new JSONObject();try{put(o,"network",network(connectivity.getNetworkCapabilities(connectivity.getActiveNetwork())));}catch(SecurityException e){put(o,"network_reason","ACCESS_NETWORK_STATE denied");}
    TelephonyManager tm=(TelephonyManager)getSystemService(TELEPHONY_SERVICE);
    if(tm==null){put(o,"data_network_type",unavailable("Android enum","No telephony service"));}
    else{
      try{if(checkSelfPermission(Manifest.permission.READ_PHONE_STATE)==PackageManager.PERMISSION_GRANTED)put(o,"data_network_type",metric(tm.getDataNetworkType(),"Android TelephonyManager network-type enum"));else put(o,"data_network_type",unavailable("Android enum","READ_PHONE_STATE not granted;no identifiers requested"));}catch(RuntimeException e){put(o,"data_network_type",unavailable("Android enum",e.getClass().getSimpleName()));}
      try{SignalStrength s=tm.getSignalStrength();JSONArray signals=new JSONArray();if(s!=null)for(CellSignalStrength x:s.getCellSignalStrengths()){JSONObject v=new JSONObject();put(v,"reported_class",x.getClass().getSimpleName());put(v,"dbm",metric(x.getDbm(),"dBm"));put(v,"level",metric(x.getLevel(),"Android ordinal0to4"));signals.put(v);}put(o,"reported_signals",signals);if(s==null||signals.length()==0)put(o,"signal_reason","No cached reported signal;not zero signal or proof of no connectivity");put(o,"signal_freshness","Latest cached OS value;physical sample age unavailable");}catch(RuntimeException e){put(o,"signal_reason",e.getClass().getSimpleName());}
    }
    BatteryManager bm=(BatteryManager)getSystemService(BATTERY_SERVICE);if(bm==null)put(o,"battery_reason","No battery service;missing diagnostics,not zero consumption");
    if(bm!=null){put(o,"battery_current_now",metric(bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW),"uA_net_battery_positive_charging"));put(o,"battery_charge_counter",metric(bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),"uAh"));put(o,"battery_energy_counter",metric(bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER),"nWh"));}
    Intent battery=registerReceiver(null,new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
    if(battery!=null){put(o,"battery_voltage",metric(battery.getIntExtra(BatteryManager.EXTRA_VOLTAGE,Integer.MIN_VALUE),"mV"));put(o,"battery_temperature",metric(battery.getIntExtra(BatteryManager.EXTRA_TEMPERATURE,Integer.MIN_VALUE),"0.1_C"));put(o,"battery_level",metric(battery.getIntExtra(BatteryManager.EXTRA_LEVEL,Integer.MIN_VALUE),"level_units"));put(o,"battery_scale",metric(battery.getIntExtra(BatteryManager.EXTRA_SCALE,Integer.MIN_VALUE),"level_units"));}
    put(o,"process_cpu_elapsed",metric(android.os.Process.getElapsedCpuTime(),"ms_process_CPU"));Debug.MemoryInfo mem=new Debug.MemoryInfo();Debug.getMemoryInfo(mem);put(o,"process_total_pss",metric(mem.getTotalPss(),"KiB"));put(o,"battery_limitation","OS battery diagnostics,not calibrated device/component/task energy or measured watts");return o;
  }
  private void localTask(boolean compute){
    if(!recording){status.setText("Start a separately authorized local capture before a task.");return;}
    long released=SystemClock.elapsedRealtimeNanos();if(!budget.task(compute?"sum":"document",released)){status.setText("Task already attempted or acceptance window expired;no repeat permitted.");return;}String expected=compute?"55\n":"Draft note\nEdited locally.\n";String actual="";boolean reopened=false;String failure=null;
    try{
      if(compute){int sum=0;for(int i=1;i<=10;i++)sum+=i;actual=sum+"\n";}
      else{File f=new File(getFilesDir(),"original-benchmark-note.txt");try(FileOutputStream s=new FileOutputStream(f)){s.write(expected.getBytes("UTF-8"));s.getFD().sync();}try(FileInputStream s=new FileInputStream(f)){byte[] b=new byte[128];int n=s.read(b);actual=new String(b,0,n,"UTF-8");reopened=true;}}
    }catch(Exception e){failure=e.getClass().getSimpleName();}
    JSONObject o=new JSONObject();put(o,"task_id",compute?"original-java-sum":"original-file-edit");put(o,"released_ns",released);put(o,"finished_ns",SystemClock.elapsedRealtimeNanos());put(o,"clock_domain","Android_elapsedRealtimeNanos_boot");put(o,"expected_sha256",hash(expected));put(o,"observed_output_sha256",hash(actual));put(o,"correct",expected.equals(actual) && failure==null);put(o,"durable_reopened",reopened);put(o,"failure",failure);put(o,"deadline_after_release_ns",5000000000L);put(o,"scope","Original executable benchmark,not production WAIKE/DeviceOS app or power-cut durability");emit("task_result",o);
  }
  private String hash(String s){try{byte[] h=MessageDigest.getInstance("SHA-256").digest(s.getBytes("UTF-8"));StringBuilder b=new StringBuilder();for(byte x:h)b.append(String.format(Locale.ROOT,"%02x",x&255));return b.toString();}catch(Exception e){throw new IllegalStateException(e);}}
  @Override protected void onActivityResult(int q,int result,Intent data){super.onActivityResult(q,result,data);if(q==11&&result==RESULT_OK&&data!=null&&trace!=null){try(InputStream in=new FileInputStream(trace);OutputStream out=getContentResolver().openOutputStream(data.getData())){byte[] b=new byte[4096];int n;while((n=in.read(b))!=-1)out.write(b,0,n);status.setText("Exported manually;capture still requires source/rights/clock/evidence review.");}catch(Exception e){status.setText("Export failed: "+e.getClass().getSimpleName());}}}
}
