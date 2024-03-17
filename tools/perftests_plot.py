import sys
import os
import subprocess
import json
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    print("No benchmark selected")
    print("Usage: python perftests_plot.py [path_to_benchmark_binary]")
else:
    test_name = sys.argv[1]

    contents = subprocess.run([test_name, "--hpx:verbose_bench"], capture_output=True)

    # print(contents)
    json_obj = json.loads(contents.stdout.decode('utf-8'))

    test_names = []
    samples = []

    for test in json_obj["outputs"]:
        test_names.append(test["name"] + "," + test["executor"])
        samples.append(test["series"])

    fig = plt.figure()
    ax = fig.add_subplot()
    bp = ax.boxplot(samples)

    ax.set_xticklabels(test_names)

    plt.show()


    
    