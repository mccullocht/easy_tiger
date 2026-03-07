import json
import re
import sys

def parse_duration(time_str):
    # Matches patterns like "24.7ms", "1.56µs", "555µs", "4.88ms"
    # tracing-subscriber format: https://docs.rs/tracing-subscriber/latest/tracing_subscriber/fmt/time/struct.SystemTime.html
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
    if len(sys.argv) < 2:
        print("Usage: python parse_trace.py <path_to_trace.json>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    
    totals = {
        "insert_batch": 0.0,
        "merge_centroid": 0.0,
        "split_centroid": 0.0,
        "partition_centroid": 0.0,
        "delete_head_centroid": 0.0,
        "insert_head_centroids": 0.0,
        "split_reassignments": 0.0,
        "nearby_reassignments": 0.0,
        "write_reassignments": 0.0,
    }
    
    counts = {
        "insert_batch": 0,
        "merge_centroid": 0,
        "split_centroid": 0,
        "partition_centroid": 0,
        "delete_head_centroid": 0,
        "insert_head_centroids": 0,
        "split_reassignments": 0,
        "nearby_reassignments": 0,
        "write_reassignments": 0,
    }
    
    with open(filepath, 'r') as f:
        for line in f:
            if not line.strip():
                continue
                
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            # We only care about span close events which contain the duration
            if entry.get("fields", {}).get("message") != "close":
                continue
                
            span_name = entry.get("span", {}).get("name")
            
            if span_name in totals:
                time_busy = entry.get("fields", {}).get("time.busy", "0s")
                duration = parse_duration(time_busy)
                
                totals[span_name] += duration
                counts[span_name] += 1
                
    print(f"{'Trace Spans':<25} | {'Total time.busy (s)':<20} | {'Count':<10} | {'Avg (ms)':<15}")
    print("-" * 70)
    for name in totals:
        count = counts[name]
        avg = (totals[name] / count * 1000) if count > 0 else 0
        print(f"{name:<25} | {totals[name]:<20.4f} | {count:<10} | {avg:<15.4f}")

if __name__ == '__main__':
    main()
