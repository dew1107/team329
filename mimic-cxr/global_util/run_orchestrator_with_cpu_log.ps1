# ==================================
# run_orchestrator_with_cpu_log.ps1
# (CPU/RAM Logger for orchestrator.py)
# ==================================

Write-Output "Script start: Starting Orchestrator script and CPU/RAM logging."
Write-Output "All output will be saved to the .log file."

# Task 1: Start CPU/RAM monitoring (background job)
Write-Output "Starting CPU/RAM monitoring job in the background (5-second interval)."

$resJob = Start-Job -ScriptBlock {
    while ($true) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        try {
            $cpuUtil = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue
            $memAvail = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue

            Write-Output "[$timestamp] RESOURCE_LOG: CPU: $($cpuUtil.ToString('F1')) %, RAM_Available: $memAvail MB"
        } catch {
            Write-Output "[$timestamp] Failed to run Get-Counter."
        }
        Start-Sleep -Seconds 5
    }
}

# Task 2: Run main Orchestrator script
# (orchestrator.py가 프로젝트 루트에 있다고 가정)
Write-Output "Starting main Orchestrator script (orchestrator.py)..."

try {
    # -u: ensures unbuffered output from Python
    # ==========================================================
    # (필요시 --metric, --K_img 등 인수를 뒤에 추가)
    python -u ./global/orchestrator.py --metric macro_auroc --K_img 32 --K_txt 32 --d_model 256
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ==========================================================

} catch {
    Write-Output "A critical error occurred during the script: $($_.Exception.Message)"
} finally {
    Write-Output "Orchestrator script complete. Stopping resource monitoring job."
    Stop-Job $resJob
    Receive-Job $resJob
    Remove-Job $resJob
}

Write-Output "All script execution finished."