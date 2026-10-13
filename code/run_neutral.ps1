$week1 = 'D:\KimiData\kimi\tasks\2026-10-02\11-34-47-0c874f62\week1'
$py    = 'C:\Users\XMM010\AppData\Local\Python\pythoncore-3.14-64\python.exe'
Set-Location $week1
$seeds   = 17, 29, 43
$methods = 'SAS-Tool', 'SAS-SC', 'MAS-Deliberate', 'Exchange-then-Decide'
foreach ($s in $seeds) {
  foreach ($m in $methods) {
    $safe = ($m -replace '-', '_')
    Start-Process -FilePath $py `
      -ArgumentList "step22_llm_neutral.py --method `"$m`" --seed $s" `
      -WorkingDirectory $week1 `
      -WindowStyle Minimized `
      -RedirectStandardOutput "$week1\neu_${safe}_s${s}.log" `
      -RedirectStandardError  "$week1\neu_${safe}_s${s}_err.log"
    Start-Sleep -Seconds 3
  }
}
Write-Host '12 workers launched. Progress: line count of data\formal_llm_neutral.jsonl (target 2880). Re-run this script to resume.'
