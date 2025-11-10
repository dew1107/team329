# ==================================
# run_kd_with_cpu_log.ps1
# (CPU/RAM Logger for repr_kd.py)
# ==================================

Write-Output "Script start: Starting KD script and CPU/RAM logging."
Write-Output "All output will be saved to the .log file."

# Task 1: Start CPU/RAM monitoring (background job)
Write-Output "Starting CPU/RAM monitoring job in the background (5-second interval)."

$resJob = Start-Job -ScriptBlock {
    # This loop repeats until Stop-Job is called
    while ($true) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        try {
            # Get Total CPU Usage (%)
            $cpuUtil = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue
            # Get Available Memory (MB)
            $memAvail = (Get-Counter '\Memory\Available MBytes').CounterSamples[0].CookedValue

            # [2025-11-01 12:00:00] RESOURCE_LOG: CPU: 85.5 %, RAM_Available: 12050 MB
            Write-Output "[$timestamp] RESOURCE_LOG: CPU: $($cpuUtil.ToString('F1')) %, RAM_Available: $memAvail MB"

        } catch {
            Write-Output "[$timestamp] Failed to run Get-Counter."
        }

        # Wait 5 seconds
        Start-Sleep -Seconds 5
    }
}

# Task 2: Run main KD script
# (repr_kd.py가 프로젝트 루트에 있다고 가정)
# (필요시 --alpha, --kteach 등 인수를 추가하세요)
Write-Output "Starting main KD script (repr_kd.py)..."

try {
    # -u: ensures unbuffered output from Python
    # repr_kd.py 파일의 위치에 맞게 경로를 수정하세요 (예: ./local/repr_kd.py)
    python -u global\repr_kd.py

} catch {
    Write-Output "A critical error occurred during the script: $($_.Exception.Message)"
} finally {
    # When the script is finished, stop the background job
    Write-Output "KD script complete. Stopping resource monitoring job."
    Stop-Job $resJob

    # Get any remaining logs from the buffer
    Receive-Job $resJob

    # Clean up the job
    Remove-Job $resJob
}

Write-Output "All script execution finished."