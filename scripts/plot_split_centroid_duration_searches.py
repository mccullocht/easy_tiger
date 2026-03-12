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
    parser = argparse.ArgumentParser(description="Plot duration and searches of split_centroid trace events over time.")
    parser.add_argument("trace_json", help="Path to trace.json")
    parser.add_argument("--png", help="Output PNG filename", default=None)
    parser.add_argument("--show", action="store_true", help="Show interactively")
    args = parser.parse_args()

    trace_file = args.trace_json
    
    target_event = "split_centroid"

    # List of (timestamp, duration_secs, searches)
    points = []
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
            if span_name != target_event:
                continue

            time_busy_str = doc.get("fields", {}).get("time.busy", "0s")
            duration_s = parse_duration(time_busy_str)
            
            # Extract number of searches from span fields
            searches = doc.get("span", {}).get("searches", 0)
            
            if searches == 0:
                # also check fields
                searches = doc.get("fields", {}).get("searches", 0)

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
            points.append((elapsed, duration_s * 1000.0, searches)) # Store duration in ms
            
    if not points:
        print("No matching events found in trace.", file=sys.stderr)
        sys.exit(1)

    # Base PNG filename prefix
    base_png = args.png
    if base_png:
        out_png = base_png
    else:
        name = os.path.splitext(trace_file)[0]
        ext = ".png"
        out_png = f"{name}_{target_event}_duration_searches{ext}"

    gp_script = []
    
    if args.png or not args.show: # Default to png generation if not just showing
        gp_script.append("set terminal pngcairo size 1024,768 enhanced font 'Verdana,10'")
        gp_script.append(f"set output '{out_png}'")
    
    gp_script.append(f"set title 'Event {target_event}: Duration and Searches Over Time'")
    gp_script.append("set xlabel 'Time (seconds)'")
    
    # We need two y-axes
    gp_script.append("set ylabel 'Duration (ms)'")
    gp_script.append("set y2label 'Searches'")
    gp_script.append("set ytics nomirror")
    gp_script.append("set y2tics nomirror")
    
    gp_script.append("set key outside right top")
    
    gp_script.append("$data << EOD")
    for elapsed, duration_ms, searches in points:
        gp_script.append(f"{elapsed:.6f} {duration_ms:.6f} {searches}")
    gp_script.append("EOD")
    
    # Plot with lines
    gp_script.append(f"plot $data using 1:2 with lines axes x1y1 title 'Duration (ms)', \\")
    gp_script.append(f"     $data using 1:3 with lines axes x1y2 title 'Searches'")
    
    gnuplot_input = "\n".join(gp_script)
    
    if not args.png and not args.show:
         print(f"--- Gnuplot input for {target_event} ---")
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
            print(f"Error running gnuplot for {target_event}: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == '__main__':
    main()
