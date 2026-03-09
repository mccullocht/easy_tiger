#!/usr/bin/env python3
import sys
import json
import os
import subprocess
import re
from datetime import datetime
from collections import defaultdict

def parse_duration(time_str):
    match = re.match(r'^([\d\.]+)\s*([a-zA-Zµ]+)$', time_str)
    if not match:
        return 0.0
        
    val = float(match.group(1))
    unit = match.group(2)
    
    if unit == 's':
        return val
    elif unit == 'ms':
        return val / 1000.0
    elif unit == 'µs' or unit == 'us':
        return val / 1_000_000.0
    elif unit == 'ns':
        return val / 1_000_000_000.0
    else:
        return 0.0

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Plot duration of trace events over time.")
    parser.add_argument("trace_json", help="Path to trace.json")
    parser.add_argument("--png", help="Output PNG filename", default=None)
    parser.add_argument("--show", action="store_true", help="Show interactively")
    args = parser.parse_args()

    trace_file = args.trace_json
    
    target_events = {
        "split_centroid", 
        "merge_centroid", 
        "split_reassignments", 
        "nearby_reassignments", 
        "delete_head_centroid", 
        "write_assignments",
        "write_reassignments"
    }

    # Dict of event_name -> list of (timestamp, duration_secs)
    series = defaultdict(list)
    start_time = None

    with open(trace_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            if doc.get("fields", {}).get("message") != "close":
                continue
                
            span_name = doc.get("span", {}).get("name")
            if span_name not in target_events:
                continue

            time_busy_str = doc.get("fields", {}).get("time.busy", "0s")
            duration_s = parse_duration(time_busy_str)

            ts_str = doc.get("timestamp")
            if not ts_str:
                continue

            try:
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except Exception:
                continue

            if start_time is None:
                start_time = dt

            elapsed = (dt - start_time).total_seconds()
            series[span_name].append((elapsed, duration_s * 1000.0)) # Store duration in ms
            
    if not series:
        print("No matching events found in trace.", file=sys.stderr)
        sys.exit(1)

    # Base PNG filename prefix
    base_png = args.png
    if base_png:
        # If they gave `/tmp/foo.png`, we'll make `/tmp/foo_split_centroid.png`
        name, ext = os.path.splitext(base_png)
    else:
        name = os.path.splitext(trace_file)[0]
        ext = ".png"

    for event_name, points in series.items():
        gp_script = []
        
        # Determine specific output filename
        out_png = f"{name}_{event_name}{ext}"
        
        if args.png or not args.show: # Default to png generation if not just showing
            gp_script.append("set terminal pngcairo size 1024,768 enhanced font 'Verdana,10'")
            gp_script.append(f"set output '{out_png}'")
        
        gp_script.append(f"set title 'Event Duration Over Time: {event_name}'")
        gp_script.append("set xlabel 'Time (seconds)'")
        gp_script.append("set ylabel 'Duration (ms)'")
        gp_script.append("set key outside right top")
        
        gp_script.append("$data << EOD")
        for elapsed, duration_ms in points:
            gp_script.append(f"{elapsed:.6f} {duration_ms:.6f}")
        gp_script.append("EOD")
        
        # Plot with lines
        gp_script.append(f"plot $data using 1:2 with lines title '{event_name}'")
        
        gnuplot_input = "\n".join(gp_script)
        
        if not args.png and not args.show:
             print(f"--- Gnuplot input for {event_name} ---")
             print(gnuplot_input)
             print()
        else:
            try:
                cmd = ["gnuplot"]
                if args.show:
                    cmd.append("-p")
                subprocess.run(cmd, input=gnuplot_input, text=True, check=True)
                if not args.show or args.png:
                    print(f"Generated {out_png}")
            except FileNotFoundError:
                print("Error: gnuplot is not installed or not in PATH.", file=sys.stderr)
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                print(f"Error running gnuplot for {event_name}: {e}", file=sys.stderr)
                sys.exit(1)

if __name__ == '__main__':
    main()
