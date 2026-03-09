#!/usr/bin/env python3
import sys
import json
import os
import subprocess
from datetime import datetime

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def extract_len(doc):
    # Depending on how the JSON is formatted, len might be doc["span"]["len"]
    # or doc["fields"]["len"] etc.
    flat = flatten_dict(doc)
    for k, v in flat.items():
        if k.endswith("len"):
            return int(v)
    return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_insert_batch.py <trace.json>", file=sys.stderr)
        sys.exit(1)

    trace_file = sys.argv[1]

    data_points = []
    start_time = None

    with open(trace_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            
            # Check if this is an insert_batch event. We typically look for "close" events for spans to avoid double counting.
            # But the user might just want the count from any event associated with the span.
            # Let's count "close" events to represent the completion of the batch.
            is_insert_batch = False
            span = doc.get("span", {})
            if isinstance(span, dict) and span.get("name") == "insert_batch":
                is_insert_batch = True
            elif doc.get("name") == "insert_batch": # Fallback
                is_insert_batch = True
                
            if not is_insert_batch:
                continue

            # Only count 'close' if we're dealing with span events
            if "span" in doc and doc.get("fields", {}).get("message") != "close":
                continue

            ts_str = doc.get("timestamp")
            if not ts_str:
                continue

            try:
                # Tracing timestamps are often like "2023-10-18T20:47:32.446864Z"
                dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            except Exception:
                continue

            if start_time is None:
                start_time = dt

            elapsed = (dt - start_time).total_seconds()
            
            batch_len = extract_len(doc)
            if batch_len is None:
                batch_len = 0 # Fallback

            data_points.append((elapsed, batch_len))
            
    if not data_points:
        print("No insert_batch events found in trace.", file=sys.stderr)
        sys.exit(1)

    cumulative_batches = 0
    cumulative_vectors = 0
    
    png_file = os.path.splitext(trace_file)[0] + '.png'
    
    gp_script = []
    gp_script.append("set terminal pngcairo size 1024,768 enhanced font 'Verdana,10'")
    gp_script.append(f"set output '{png_file}'")
    gp_script.append("set title 'Cumulative Inserted Batches and Vectors over Time'")
    gp_script.append("set xlabel 'Time (seconds)'")
    gp_script.append("set ylabel 'Cumulative Count (Batches)'")
    gp_script.append("set y2label 'Cumulative Count (Vectors)'")
    gp_script.append("set ytics nomirror")
    gp_script.append("set y2tics")
    gp_script.append("set key left top")
    gp_script.append("$data << EOD")
    gp_script.append("# Time(s) Cum_Batches Cum_Vectors")
    
    # Starting point at 0
    gp_script.append(f"0.000000 0 0")
    
    for elapsed, batch_len in data_points:
        cumulative_batches += 1
        cumulative_vectors += batch_len
        gp_script.append(f"{elapsed:.6f} {cumulative_batches} {cumulative_vectors}")
        
    gp_script.append("EOD")
    gp_script.append("plot $data using 1:2 with lines axes x1y1 title 'Cumulative Batches', \\")
    gp_script.append("     $data using 1:3 with lines axes x1y2 title 'Cumulative Vectors'")
    
    gnuplot_input = "\n".join(gp_script)
    
    try:
        subprocess.run(["gnuplot"], input=gnuplot_input, text=True, check=True)
        print(f"Generated {png_file}")
    except FileNotFoundError:
        print("Error: gnuplot is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running gnuplot: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
