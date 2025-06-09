import numpy as np
import pandas as pd
from bayesmixpy import build_bayesmix, run_mcmc


dp = """
gamma_prior {
  totalmass_prior {
    shape: 4.0
    rate: 2.0
  }
}
"""

algo = """
algo_id: "Neal2"
rng_seed: 20201124
iterations: 2000
burnin: 1000
init_num_clusters: 10
"""

def compute_waic(L: np.ndarray) -> float:
    M, n = L.shape  # M = number of iterations, n = number of datapoints
    lppd = 0.0  # log pointwise predictive density
    p_waic = 0.0  # effective number of parameters

    for j in range(n):
        m = np.max(L[:, j])
        lppd_j = m + np.log(np.mean(np.exp(L[:, j] - m)))
        lppd += lppd_j

        p_waic += np.var(L[:, j])

    return -2 * (lppd - p_waic)

def run_one(data):
    m = np.mean(data)
    v = np.var(data)
    
    g0 = f"""fixed_values {{
    mean: {m}
    var_scaling: 0.1
    shape: 2.0
    scale: {v/2}
    }}"""  
    
    log_liks, _, _, _, _= run_mcmc(
        "NNIG", "DP", data, g0, dp, algo, data,
        return_clusters=False, return_num_clusters=False,
        return_best_clus=False)
    
    waic = compute_waic(log_liks)
    
    print("DPM: ", waic)
    
    return waic


if __name__ == "__main__":
    
    rows = []
    
    print("EURODIST")
    data = np.loadtxt("data/eurodist.csv")
    waic = run_one(data)
    rows.append(("eurodist", waic))
    

    print("GDP")
    data = np.loadtxt("data/gdp.csv")
    waic = run_one(data)
    rows.append(("gdp", waic))

    print("CENSUS")
    data = np.loadtxt("data/census.csv")
    waic = run_one(data)
    rows.append(("census", waic))

    print("TWITTER")
    data = np.loadtxt("data/twitter_friends.csv")
    waic = run_one(data)
    rows.append(("twitter", waic))

    print("INCOME")
    data = np.loadtxt("data/income.csv")
    waic = run_one(data)
    rows.append(("GDP", waic))

    df = pd.DataFrame(rows)
    df.to_csv("data/real_data_dpm_out.csv")

