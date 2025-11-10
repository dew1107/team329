# ==================================
# run_all_with_gpu_log.ps1 (English Ver.)
# ==================================

Write-Output "Script start: Starting training and GPU logging."
Write-Output "All output will be saved to the .log file."

# Task 1: Start GPU monitoring (background job)
Write-Output "Starting GPU monitoring job in the background (30-second interval)."

$gpuJob = Start-Job -ScriptBlock {
    # This loop repeats until Stop-Job is called
    while ($true) {
        $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        try {
            # 1. 쿼리할 항목 추가:
            #   - utilization.gpu : GPU 코어 사용률 (%)
            #   - memory.used     : 사용 중인 VRAM (MiB)
            #   - memory.total    : 전체 VRAM (MiB)
            #   - power.draw      : 현재 전력 소모 (W)
            #   - temperature.gpu : GPU 온도 (C)
            $gpuData = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits

            # 2. 출력 포맷 변경
            # $gpuData는 "92, 4096, 8192, 150.5, 75" 같은 문자열이 됩니다.
            $parts = $gpuData -split ', '

            $util     = $parts[0].Trim()
            $memUsed  = $parts[1].Trim()
            $memTotal = $parts[2].Trim()
            $power    = $parts[3].Trim()
            $temp     = $parts[4].Trim()

            # [2025-11-01 12:00:00] GPU_LOG: Util: 92 %, Mem: 4096/8192 MiB, Power: 150.5 W, Temp: 75 C
            Write-Output "[$timestamp] GPU_LOG: Util: $util %, Mem: $memUsed/$memTotal MiB, Power: $power W, Temp: $temp C"

        } catch {
            Write-Output "[$timestamp] Failed to run nvidia-smi."
        }

        # 30초 대기 (이 값은 그대로 30초를 사용합니다)
        Start-Sleep -Seconds 30
    }
}

# Task 2: Run main training script (Clients 1-20)
Write-Output "Starting main training job (Client 1 ~ 20)."

try {
    (1..20) | ForEach-Object {
        # ==========================================================
        # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
        # Updated part:
        # Before starting the next client, get all buffered GPU logs.
        Receive-Job $gpuJob
        # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
        # ==========================================================

        Write-Output "[Orchestrator] Starting training for Client $_..."
        # -u: ensures unbuffered output from Python
        python -u ./local/train_local.py --client_id $_
    }
} catch {
    Write-Output "A critical error occurred during training: $($_.Exception.Message)"
} finally {
    # When training is finished (success or fail), stop the background job
    Write-Output "All training jobs complete. Stopping GPU monitoring job."
    Stop-Job $gpuJob

    # ==========================================================
    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
    # Updated part:
    # Get any remaining logs from the buffer one last time.
    Receive-Job $gpuJob
    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
    # ==========================================================

    # Clean up the job
    Remove-Job $gpuJob
}

Write-Output "All script execution finished."