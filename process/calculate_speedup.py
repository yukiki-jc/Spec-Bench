
def cal_exp_speedup(alpha, gamma, c): 
    numerator = 1 - alpha ** (gamma + 1)
    denominator = (1 - alpha) * (c * gamma + 1)
    return numerator / denominator

alpha = 0.7 
import numpy as np 
c = 0.1
c_dict = {}
for c in np.arange(0.1, 0.4, 0.05):
    c_dict[c] = []
    # for alpha in np.arange(0.5, 0.7, 0.05): 
    for gamma in range(1, 11):
        speedup = cal_exp_speedup(alpha, gamma, c)
        c_dict[c].append(speedup)
        # print(f"alpha: {alpha}, gamma: {gamma}, c: {c}, speedup: {speedup}")
import matplotlib.pyplot as plt
for c, speedup_list in c_dict.items():
    best_dl = np.argmax(speedup_list) + 1
    plt.plot(range(1, 11), speedup_list, label=f"c={c: .2f}, best DL={best_dl}", marker='o')
plt.xlabel("Draft Length")
plt.ylabel("Speedup")
plt.xticks(range(1, 11))
plt.legend()
plt.savefig("speedup-vs-dl.png")