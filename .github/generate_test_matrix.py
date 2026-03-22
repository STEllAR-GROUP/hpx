#!/usr/bin/env python3
# Copyright (c) 2026 The STE||AR Group
# Copyright (c) 2026 Surya Teja
# SPDX-License-Identifier: BSL-1.0
# Distributed under the Boost Software License, Version 1.0. (See accompanying
# file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

import sys
import json
import re
import math

def generate_matrix(ctest_output, num_buckets):
    tests = []
    pattern = re.compile(r"^\s*Test\s*#\d+:\s*(.+)$")
    
    for line in ctest_output.splitlines():
        match = pattern.match(line)
        if match:
            test_name = match.group(1).strip()
            tests.append(test_name)
    
    tests.sort()
    
    if not tests:
        return {"include": []}
    
    total_tests = len(tests)
    bucket_size = math.ceil(total_tests / num_buckets)
    
    matrix = {"include": []}
    
    for i in range(num_buckets):
        start = i * bucket_size
        end = min((i + 1) * bucket_size, total_tests)
        
        if start >= total_tests:
            break
            
        bucket_tests = tests[start:end]
        
        test_regex = "^(" + "|".join(re.escape(t) for t in bucket_tests) + ")$"
        
        # distributed tests have .distributed.<parcelport>. in their name which
        # is not in the target
        bucket_targets = []
        for t in bucket_tests:
            target = re.sub(r'\.distributed\.(tcp|mpi|lci|lcw|gasnet)\.', '.', t)
            bucket_targets.append(target)
            
        unique_targets = sorted(list(set(bucket_targets)))
        targets_str = " ".join(unique_targets)
        
        matrix["include"].append({
            "id": i + 1,
            "name": f"tests-{i+1:02d}",
            "tests": test_regex,
            "targets": targets_str,
            "count": len(bucket_tests)
        })
        
    return matrix

if __name__ == "__main__":
    try:
        input_data = sys.stdin.read()
        num_jobs = 10
        if len(sys.argv) > 1:
            num_jobs = int(sys.argv[1])
            
        matrix = generate_matrix(input_data, num_jobs)
        print(json.dumps(matrix))
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
