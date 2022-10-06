import matplotlib.pyplot as plt
import numpy as onp
import os

# Latex style plot
plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})


def plot_results():
    root_path = f"modules/fem/apps/top_opt/data/"

    plate_obj = onp.load(os.path.join(root_path, f"numpy/plate_outputs.npy")).reshape(-1)
    bracket_obj = onp.load(os.path.join(root_path, f"numpy/bracket_outputs.npy")).reshape(-1)
    plate_obj = plate_obj
    data = [plate_obj, bracket_obj]

    cases = ['plate', 'bracket']
    for i in range(len(cases)):
        print(data[i][-1])
        plt.figure(figsize=(8, 6))
        # plt.plot(steps, data[i], linestyle='-', marker='o', markersize=10, linewidth=2, color='black')
        plt.plot(onp.arange(len(data[i])), data[i], linestyle='-', linewidth=2, color='black')
        plt.xlabel(r"Optimization step", fontsize=20)
        plt.ylabel(r"Compliance [$\mu$J]", fontsize=20)
        plt.tick_params(labelsize=20)
        plt.tick_params(labelsize=20)
        plt.savefig(os.path.join(root_path, f'pdf/{cases[i]}_obj.pdf'), bbox_inches='tight')


if __name__=="__main__":
    plot_results()
    plt.show()
