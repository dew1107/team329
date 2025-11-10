# ==================================
# run_prep_with_gpu_log.ps1
# (GPU Logger for prep_clients.py)
# .\global\run_prep_with_gpu_log.ps1 2>&1 | Tee-Object -FilePath "prep_run.log"
# ==================================

Write-Output "Script start: Starting Prep-Clients script and GPU logging."
Write-Output "All output will be saved to the .log file."

# Task 1: Start GPU monitoring (background job)
Write-Output "Starting GPU monitoring job in the background (30-second interval)."

# (이전에 수정한 GPU 상세 로그 버전)
$gpuJob = Start-Job -ScriptBlock {
    while ($true) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        try {
            $gpuData = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits
            $parts = $gpuData -split ', '
            $util     = $parts[0].Trim()
            $memUsed  = $parts[1].Trim()
            $memTotal = $parts[2].Trim()
            $power    = $parts[3].Trim()
            $temp     = $parts[4].Trim()
            Write-Output "[$timestamp] GPU_LOG: Util: $util %, Mem: $memUsed/$memTotal MiB, Power: $power W, Temp: $temp C"
        } catch {
            Write-Output "[$timestamp] Failed to run nvidia-smi."
        }
        Start-Sleep -Seconds 30 # 30초 간격
    }
}

# Task 2: Run main Prep-Clients script
# (prep_clients.py가 프로젝트 루트에 있다고 가정)
Write-Output "Starting main Prep-Clients script (prep_clients.py)..."

try {
    # -u: ensures unbuffered output from Python
    # ==========================================================
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # 이 부분만 수정합니다:
    # (prep_clients.py는 1-20을 인자로 받으므로 루프가 필요 없음)
    python -u .\global\prep_clients.py --cids "7-20" # 이전에 7까지 끝내놓음
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ==========================================================

} catch {
    Write-Output "A critical error occurred during the script: $($_.Exception.Message)"
} finally {
    Write-Output "Prep-Clients script complete. Stopping resource monitoring job."
    Stop-Job $gpuJob
    Receive-Job $gpuJob
    Remove-Job $gpuJob
}

Write-Output "All script execution finished."