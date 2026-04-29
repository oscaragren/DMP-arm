import csv
from time import monotonic_ns, sleep
from hardware.can.can_socketcan import SocketCANInterface

def send_robot_commands(robot_commands, command_period_ms=20, log_path="timing_send.csv"):
    period_ns = int(command_period_ms * 1_000_000)
    logs = []

    start_ns = monotonic_ns()
    previous_send_start_ns = None


    for i, command in enumerate(robot_commands):
        planned_ns = start_ns + i * period_ns
        
        now_ns = monotonic_ns()
        if now_ns < planned_ns:
            sleep((planned_ns-now_ns)/1_000_000_000)
        
        actual_start_ns = monotonic_ns()

        success = True
        for can_id, data in command:
            ok = can_interface.send(can_id, data)
            success = success and ok

        actual_end_ns = monotonic_ns()

        send_duration_ms = (actual_end_ns - actual_start_ns) / 1_000_000
        schedule_error_ms = (actual_start_ns - planned_ns) / 1_000_000

        if previous_send_start_ns is None:
            period_ms = None
        else:
            period_ms = (actual_start_ns - previous_send_start_ns) / 1_000_000

        deadline_miss = actual_end_ns > planned_ns + period_ns

        logs.append({
            "index": i,
            "planned_ns": planned_ns,
            "actual_start_ns": actual_start_ns,
            "actual_end_ns": actual_end_ns,
            "send_duration_ms": send_duration_ms,
            "schedule_error_ms": schedule_error_ms,
            "period_ms": period_ms,
            "deadline_miss": deadline_miss,
            "success": success,
        })

        previous_send_start_ns = actual_start_ns

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=logs[0].keys())
        writer.writeheader()
        writer.writerows(logs)

    return all(row["success"] for row in logs)