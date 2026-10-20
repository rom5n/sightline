# Total memory of the running app: shell + its WebView2 processes + the Python engine.
# Usage: powershell -File e2e/mem.ps1
$all = Get-CimInstance Win32_Process
$root = $all | Where-Object { $_.Name -eq 'sightline.exe' } | Select-Object -First 1
if (-not $root) { Write-Output 'sightline.exe is not running'; exit 1 }
$ids = @($root.ProcessId)
do {
  $more = $all | Where-Object { $ids -contains $_.ParentProcessId -and $ids -notcontains $_.ProcessId } | ForEach-Object { $_.ProcessId }
  $ids += $more
} while ($more)
# WebView2 browser processes are started by the shell's WebView2 loader and parent to sightline.exe
# "Working Set - Private" is the Memory column in Task Manager; PrivateMB is committed private bytes
$pws = @{}
foreach ($s in (Get-CimInstance Win32_PerfRawData_PerfProc_Process)) { $pws[[int]$s.IDProcess] = $s.WorkingSetPrivate }
$rows = foreach ($id in $ids) {
  $p = Get-Process -Id $id -ErrorAction SilentlyContinue
  if ($p) { [pscustomobject]@{ Name = $p.ProcessName; Id = $id; PrivateWorkingSetMB = [math]::Round($pws[[int]$id] / 1MB); WorkingSetMB = [math]::Round($p.WorkingSet64 / 1MB); PrivateMB = [math]::Round($p.PrivateMemorySize64 / 1MB) } }
}
$rows | Sort-Object PrivateMB -Descending | Format-Table -AutoSize | Out-String | Write-Output
$ws = ($rows | Measure-Object WorkingSetMB -Sum).Sum
$pv = ($rows | Measure-Object PrivateMB -Sum).Sum
$pw = ($rows | Measure-Object PrivateWorkingSetMB -Sum).Sum
Write-Output "TOTAL processes=$($rows.Count) privateWorkingSetMB=$pw workingSetMB=$ws privateBytesMB=$pv"
