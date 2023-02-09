import numpy as np
import matplotlib.pyplot as plt

def main():
    expert_accs = []
    for task in range(5, 20):
        my_data = np.genfromtxt(f'src/{task}.txt', delimiter=' ')
        expert_accs.append(np.mean(my_data, axis=0))
    expert_accs = np.stack(expert_accs)


    accs = []
    for task in range(5, 20):
        my_data = np.genfromtxt(f'src/{task}_ensembled.txt', delimiter=' ')
        accs.append(np.mean(my_data, axis=0))
    accs = np.expand_dims(np.stack(accs), axis=0)
    print(accs)

    vegetables = ["Expert 1", "Expert 2", "Expert 3", "Expert 4",
               "Expert 5", "BERG"]
    farmers = ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"]
    expert_accs = np.transpose(expert_accs)
    expert_accs = np.concatenate((expert_accs, accs), axis=0)
    expert_accs = expert_accs - expert_accs[:-1].mean(axis=0)
    expert_accs*=100

    fig, ax = plt.subplots()
    im = ax.imshow(expert_accs, cmap="RdYlGn", vmin=-8, vmax=8)

    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(farmers)), labels=farmers, fontsize=16)
    ax.set_yticks(np.arange(len(vegetables)), labels=vegetables, fontsize=16)
    bar = ax.figure.colorbar(im,
                             ax=ax,
                             shrink=0.5, ticks=[8, 4, 0, -4, -8])
    bar.ax.tick_params(labelsize=16)


    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #          rotation_mode="anchor")

    # for i in range(len(vegetables)):
    #     for j in range(len(farmers)):
    #         text = ax.text(j, i, round(expert_accs[i, j], 2),
    #                        ha="center", va="center", color="black")
    plt.xlabel("Task", fontsize=18)
    fig.tight_layout()
    plt.savefig("lol.png", dpi=500)
    plt.show()



if __name__ == '__main__':
    main()