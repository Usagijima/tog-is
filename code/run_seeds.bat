@echo off
cd /d D:\KimiData\kimi\tasks\2026-10-02\11-34-47-0c874f62\week1
for %%S in (29 43) do (
  for %%C in (A C S) do (
    "C:\Users\XMM010\AppData\Local\Python\pythoncore-3.14-64\python.exe" step14_formal.py --cond %%C --seed %%S >> run_seeds.log 2>&1
  )
)
echo DONE_MARKER >> run_seeds.log
