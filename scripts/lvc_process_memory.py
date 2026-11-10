"""Low-frequency /proc sampling for the native cache benchmark (no root needed)."""
import csv
import json
import os
from pathlib import Path
import subprocess
import time


FIELDS = (
    "unix_ms", "elapsed_seconds", "pid", "sample_error", "page_size_bytes",
    "VmRSS_kib", "VmHWM_kib", "VmSize_kib", "VmData_kib", "VmSwap_kib",
    "RssAnon_kib", "RssFile_kib", "Threads", "process_minor_faults", "process_major_faults",
    "MemTotal_kib", "MemAvailable_kib", "SwapTotal_kib", "SwapFree_kib", "SwapCached_kib",
    "system_pswpin_pages", "system_pswpout_pages", "system_pgmajfault",
)


def key_values(path, colon=True):
    result = {}
    for line in path.read_text().splitlines():
        fields = line.replace(":", "", 1).split() if colon else line.split()
        if len(fields) >= 2 and fields[1].isdigit():
            result[fields[0]] = int(fields[1])
    return result


def sample(pid, started, proc_root=Path("/proc")):
    row = dict(unix_ms=time.time_ns() // 1000000, elapsed_seconds=time.monotonic() - started,
               pid=pid, sample_error="", page_size_bytes=os.sysconf("SC_PAGE_SIZE"))
    errors = []
    try:
        status = key_values(proc_root / str(pid) / "status")
        for field in ("VmRSS", "VmHWM", "VmSize", "VmData", "VmSwap", "RssAnon", "RssFile"):
            if field in status:
                row[field + "_kib"] = status[field]
        if "Threads" in status:
            row["Threads"] = status["Threads"]
        # comm (field 2) can contain spaces and parentheses; count after its final ')'.
        fields = (proc_root / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        row["process_minor_faults"] = int(fields[7])  # field 10
        row["process_major_faults"] = int(fields[9])  # field 12
    except (OSError, ValueError, IndexError) as exc:
        errors.append("process: " + str(exc))
    try:
        memory = key_values(proc_root / "meminfo")
        for field in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree", "SwapCached"):
            row[field + "_kib"] = memory[field]
        vm = key_values(proc_root / "vmstat", colon=False)
        for field in ("pswpin", "pswpout", "pgmajfault"):
            row["system_" + field + ("_pages" if field != "pgmajfault" else "")] = vm[field]
    except (OSError, ValueError, KeyError) as exc:
        errors.append("system: " + str(exc))
    row["sample_error"] = "; ".join(errors)
    return row


def summarize_samples(rows):
    def values(key):
        return [float(r[key]) for r in rows if r.get(key, "") != ""]
    report = {"memory_samples": len(rows),
              "memory_sample_errors": sum(bool(r.get("sample_error")) for r in rows)}
    for source, target, operation in (
        ("VmRSS_kib", "sampled_peak_rss_kib", max),
        ("VmSwap_kib", "sampled_peak_swap_kib", max),
        ("MemAvailable_kib", "sampled_min_mem_available_kib", min),
    ):
        data = values(source)
        report[target] = operation(data) if data else ""
    for source, target in (("system_pswpin_pages", "system_swapin_bytes"),
                           ("system_pswpout_pages", "system_swapout_bytes")):
        data = values(source)
        report[target] = ((data[-1] - data[0]) * int(rows[0]["page_size_bytes"])) if data else ""
    return report


def correlate_rounds(directory, samples):
    """Sampled process peaks within each native query interval; system counters are global."""
    path = directory / "rounds.csv"
    if not path.exists():
        return
    with path.open(newline="") as stream:
        rounds = list(csv.DictReader(stream))
    output = []
    for native in rounds:
        if not native.get("start_unix_ms") or not native.get("wall_seconds"):
            continue  # A killed process can leave an incomplete final CSV line.
        start = float(native["start_unix_ms"])
        end = start + float(native["wall_seconds"]) * 1000
        inside = [s for s in samples if start <= float(s["unix_ms"]) <= end]
        # Bracketing snapshots give context for short rounds, but are NOT round-local counters.
        before = next((s for s in reversed(samples) if float(s["unix_ms"]) <= start), None)
        after = next((s for s in samples if float(s["unix_ms"]) >= end), None)
        row = {k: native.get(k, "") for k in ("L", "round", "wall_seconds", "qps", "mean_us",
                                               "minor_faults", "major_faults")}
        row.update(start_unix_ms=start, end_unix_ms=end, **summarize_samples(inside))
        for label, point in (("before", before), ("after", after)):
            for field in ("unix_ms", "VmSwap_kib", "MemAvailable_kib"):
                row[label + "_" + field] = point.get(field, "") if point else ""
        output.append(row)
    if output:
        with (directory / "memory_rounds.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)


def run_monitored(command, log, directory, interval):
    started = time.monotonic()
    samples = []
    with (directory / "memory_samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT) as process:
            try:
                while process.poll() is None:
                    row = sample(process.pid, started)
                    samples.append(row)
                    writer.writerow(row)
                    stream.flush()
                    try:
                        process.wait(timeout=interval)
                    except subprocess.TimeoutExpired:
                        pass
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    report = summarize_samples(samples)
    report.update(pid=process.pid, sample_interval_seconds=interval,
                  system_counters_scope="whole host; other processes may contribute",
                  peaks_scope="periodic samples, not guaranteed lifetime maxima")
    (directory / "memory_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    correlate_rounds(directory, samples)
    return subprocess.CompletedProcess(command, process.returncode), report
