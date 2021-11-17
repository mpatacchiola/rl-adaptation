import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
plt.grid()

total_timesteps = 3000000
env_name = "halfcheetah"

def get_xy(logs_path, verbose=True):
    logs_list = [each for each in os.listdir(logs_path) if each.endswith(".csv")]
    y_list = list()
    for log in logs_list:
        log_path = logs_path + "/" + log
        if(verbose): print("[INFO] Reading log:", log_path)
        df = pd.read_csv(log_path)
        y_list.append(df["reward"].to_numpy().squeeze())
            
    return df["step"].to_numpy().squeeze(), np.stack(y_list, axis=0).mean(0), np.stack(y_list, axis=0).std(0)


def main():
    logs_path = "/home/mpatacchiola/Desktop/tmp_rl-adaptation/ppo/halfcheetah"
    x, y_mean, y_std = get_xy(logs_path)

    #plt.errorbar(x, y_mean, yerr=y_std, ecolor="lightblue", 
    #         elinewidth=0.5, linewidth=1.0, label="PPO",
    #         linestyle="-", c="blue")
    plt.errorbar(x, y_mean, label="PPO", c="blue", linewidth=1.0, linestyle="-")
    plt.fill_between(x, y_mean-y_std, y_mean+y_std, color='lightblue', alpha=0.5)
                 
    #plt.ticklabel_format(axis='x', style='sci')
    plt.title("Reward VS Timesteps")
    plt.xlabel("Timestep")
    plt.ylabel("Reward")
    #plt.xlim(0, total_timesteps)

    #plt.xlim([0.0, 1.0])
    #plt.ylim([-3500, 1000])
    plt.legend()

    plt.savefig("./plot_reward_vs_timesteps_" + env_name + ".pdf", dpi=300)
    plt.close()
    
    
if __name__ == "__main__":
    main() 
