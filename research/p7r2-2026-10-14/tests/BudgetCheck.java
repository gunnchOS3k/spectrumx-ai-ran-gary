import com.gunnchos.research.p7r2.AcceptanceBudget;
public final class BudgetCheck{
 static int checks=0;
 static void ok(boolean b){checks++;if(!b)throw new AssertionError("Budget assertion"+checks);}
 public static void main(String[] args){AcceptanceBudget b=new AcceptanceBudget(100);ok(!b.active(99));ok(b.active(100));ok(!b.active(120000000100L));for(int i=0;i<120;i++)ok(b.snapshot(101+i));ok(!b.snapshot(300));ok(b.task("sum",300));ok(!b.task("sum",301));ok(b.task("document",302));ok(!b.task("other",303));ok(!b.task("document",120000000100L));System.out.println("PASS "+checks+" deterministic admission assertions;not Android runtime validation");}
}
