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
        "rebalance": 0.0,
        "merge": 0.0,
        "split": 0.0
    }
    
    counts = {
        "insert_batch": 0,
        "rebalance": 0,
        "merge": 0,
        "split": 0
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
                
    print(f"{'Trace Spans':<15} | {'Total time.busy (s)':<20} | {'Count':<10} | {'Avg (ms)':<15}")
    print("-" * 70)
    for name in totals:
        count = counts[name]
        avg = (totals[name] / count * 1000) if count > 0 else 0
        print(f"{name:<15} | {totals[name]:<20.4f} | {count:<10} | {avg:<15.4f}")

if __name__ == '__main__':
    main()
