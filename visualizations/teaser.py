import matplotlib.pyplot as plt

fig, ax = plt.subplots()
fig.set_size_inches(5, 3)

approaches = ['1', '2', '3']
acc = [23.9, 46.0, 51.1]
err = [0.5, 0.7, 0.8]
bar_labels = ['red', 'blue', '_red']
bar_colors = ['#ff8000', '#00cc00', '#9933ff']

w = ax.bar(approaches, acc, label=bar_labels, color=bar_colors, edgecolor='black')

ax.errorbar(approaches, acc,  yerr=err, fmt=",", color="black", capsize=3)

plt.ylim(20, 55)

plt.xticks(fontsize=22)
plt.yticks(fontsize=18)
plt.ylabel("Average accuracy (%)", fontsize=18)

plt.savefig("teaser.png", dpi=200, bbox_inches='tight')
