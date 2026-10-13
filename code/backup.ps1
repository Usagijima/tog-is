# backup.ps1 - project backup (ASCII-only source; Chinese paths resolved via regex escapes)
# Copies the whole experiment (code + data + paper draft) to a timestamped folder
# under <Desktop>\广州大学黄埔论文\AI博弈\项目备份\
$ErrorActionPreference = "Stop"
$src = "D:\KimiData\kimi\tasks\2026-10-02\11-34-47-0c874f62"
# destination priority: OneDrive (cloud-synced) -> D:\ fallback
$cand = @($env:OneDrive, "C:\Users\XMM010\OneDrive") | Where-Object { $_ -and (Test-Path $_) } | Select-Object -First 1
if ($cand) {
  $bakRoot = Join-Path $cand ([regex]::Unescape("\u9879\u76ee\u5907\u4efd"))
} else {
  $bakRoot = "D:\project_backup"
}
New-Item -ItemType Directory -Force $bakRoot | Out-Null
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm"
$dst = Join-Path $bakRoot ("backup_" + $stamp)
New-Item -ItemType Directory -Force $dst | Out-Null

# week1 code + data, excluding caches
robocopy (Join-Path $src "week1") (Join-Path $dst "week1") /E /XD "__pycache__" "clean" "split" /NFL /NDL /NJH /NJS | Out-Null
# root documents
Copy-Item (Join-Path $src "*.md") $dst -Force

# prune: keep only the newest 4 backups
Get-ChildItem $bakRoot -Directory | Sort-Object Name -Descending |
  Select-Object -Skip 4 | Remove-Item -Recurse -Force

"$stamp backed up to $dst" | Out-File (Join-Path $bakRoot "backup.log") -Append -Encoding utf8
Write-Output "Backup done: $dst"
