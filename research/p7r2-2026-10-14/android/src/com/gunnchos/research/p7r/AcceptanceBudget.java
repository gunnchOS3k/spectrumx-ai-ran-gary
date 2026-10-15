package com.gunnchos.research.p7r2;
import java.util.HashSet;
/** Pure admission checks;Android scheduling is not hard realtime. */
public final class AcceptanceBudget {
 public static final long MAX_NS=120000000000L;
 public final long startNs;
 private int snapshots=0;
 private final HashSet<String> tasks=new HashSet<>();
 public AcceptanceBudget(long start){if(start<0)throw new IllegalArgumentException("Nonnegative boot clock");startNs=start;}
 public synchronized boolean active(long now){return now>=startNs&&now-startNs<MAX_NS;}
 public synchronized boolean snapshot(long now){if(!active(now)||snapshots>=120)return false;snapshots++;return true;}
 public synchronized boolean task(String task,long now){if(!active(now)||!("document".equals(task)||"sum".equals(task))||tasks.contains(task))return false;tasks.add(task);return true;}
}
