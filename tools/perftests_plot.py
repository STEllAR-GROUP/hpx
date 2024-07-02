import sys
import subprocess
import json
import matplotlib.pyplot as plt

if len(sys.argv) != 2:
    if len(sys.argv) == 1:
        print("No benchmark selected!")
    else:
        print("Too many arguments!")
    print("Usage: python perftests_plot.py [path_to_benchmark_binary]")
else:
    test_name = sys.argv[1]

    contents = subprocess.run([test_name, "--detailed_bench"], capture_output=True)

    json_obj = json.loads(contents.stdout.decode('utf-8'))

    test_names = []
    samples = []

    for test in json_obj["outputs"]:
        test_names.append(test["name"] + ",\n" + test["executor"])
        samples.append(test["series"])

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot()
    bp = ax.boxplot(samples, showfliers=False)
    plt.setp(ax.set_xticklabels(test_names), fontsize=7)
    plt.show()
    