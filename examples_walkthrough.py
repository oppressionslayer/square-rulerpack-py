"""The step-by-step session as a script: run it, or paste the lines into IPython one by one.
Prints what each stage hands back. (`WALKTHROUGH.md` is this script's output.)"""
import pandas as pd
import square_rulerpack_py as sq

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 60)

n = 61744934419289236901 * 94866163883639224007          # a 40-digit semiprime
w = sq.Walkthrough(n, workers=1)

print(">>> w.summary()");            print(w.summary(), "\n")
print(">>> w.multiplier().head(6)"); print(w.multiplier().head(6), "\n")
print(">>> fb = w.factor_base(); fb.head(10)"); fb = w.factor_base(); print(fb.head(10), "\n")
print(">>> fb.tail(3)");             print(fb.tail(3), "\n")
print(">>> w.family()");             print(w.family(), "\n")
print(">>> w.polynomials().head(8)"); print(w.polynomials().head(8), "\n")
print(">>> w.roots(0).head(12)");    print(w.roots(0).head(12), "\n")
print(">>> s = w.sieve(0); s");      s = w.sieve(0); print(s, "\n")
print(">>> s.acc[:16]  # the first 16 bytes of the sieve array, x = -M ... -M+15"); print(s.acc[:16], "\n")
print(">>> s.table(10)");            print(s.table(10), "\n")
print(">>> td = w.trial_divide(0); td.head(12)"); td = w.trial_divide(0); print(td.head(12), "\n")
print(">>> td.kind.value_counts()"); print(td.kind.value_counts(), "\n")
print(">>> w.collect()");            print(w.collect(), "\n")
print(">>> w.relation(0)");          print(w.relation(0), "\n")
print(">>> w.relations().head()");   print(w.relations().head(), "\n")
print(">>> w.partials().head()");    print(w.partials().head(), "\n")
print(">>> w.cycle_summary()");      print(w.cycle_summary(), "\n")
print(">>> m = w.matrix(dense=False); m.head()"); m = w.matrix(dense=False); print(m.head(), "\n")
print(">>> w.dependencies().head()"); print(w.dependencies().head(), "\n")
print(">>> w.square_root(0)");       print(w.square_root(0), "\n")
print(">>> w.factors()");            print(w.factors(), "\n")
