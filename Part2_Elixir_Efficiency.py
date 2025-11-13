import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

def load_sample(path, chunk_size=500_000, max_chunks=2):
    chunks=[]
    for i,chunk in enumerate(pd.read_csv(path, chunksize=chunk_size)):
        chunks.append(chunk)
        if i+1>=max_chunks: break
    return pd.concat(chunks, ignore_index=True)

df = load_sample("battles.csv")

cols = [
    "battleTime","arena.id","average.startingTrophies",
    "winner.elixir.average","loser.elixir.average"
]
df = df[cols].dropna().copy()
df["battleTime"] = pd.to_datetime(df["battleTime"], errors="coerce")
df = df[df["battleTime"].notnull()]

df["elixir_diff"] = df["winner.elixir.average"] - df["loser.elixir.average"]
df["patch_period"] = np.where(df["battleTime"] < pd.Timestamp("2020-12-09"), "Before","After")

bins = [0,3000,4000,5000,6000,7000,99999]
labels = ["<3k","3k-4k","4k-5k","5k-6k","6k-7k","7k+"]
df["trophy_range"] = pd.cut(df["average.startingTrophies"], bins=bins, labels=labels, include_lowest=True)

summary = pd.DataFrame({
    "avg_winner_elixir":[df["winner.elixir.average"].mean()],
    "avg_loser_elixir":[df["loser.elixir.average"].mean()],
    "median_elixir_diff":[df["elixir_diff"].median()],
    "n_matches":[len(df)]
})
summary.to_csv("part2_elixir_summary.csv", index=False)

plt.figure(figsize=(8,5))
plt.bar(["Winners","Losers"], [summary.loc[0,"avg_winner_elixir"], summary.loc[0,"avg_loser_elixir"]])
plt.ylabel("Average Elixir")
plt.title("Average Elixir Cost: Winners vs Losers")
plt.tight_layout()
plt.savefig("avg_elixir_comparison.png", dpi=300)
plt.close()

plt.figure(figsize=(9,5))
sns.histplot(data=df, x="winner.elixir.average", hue="patch_period", kde=True, bins=30, stat="density", common_norm=False)
plt.xlabel("Winner Average Elixir")
plt.ylabel("Density")
plt.title("Winner Elixir Distribution — Before vs After Patch")
plt.tight_layout()
plt.savefig("elixir_distribution_patch.png", dpi=300)
plt.close()

plt.figure(figsize=(9,5))
sns.histplot(df["elixir_diff"], bins=35, kde=True)
plt.axvline(0, ls="--", c="red")
plt.xlabel("Elixir Difference (Winner − Loser)")
plt.ylabel("Matches")
plt.title("Elixir Advantage Histogram")
plt.tight_layout()
plt.savefig("elixir_difference_hist.png", dpi=300)
plt.close()

sample_n = min(5000, len(df))
plt.figure(figsize=(6,6))
sns.scatterplot(
    data=df.sample(sample_n, random_state=42),
    x="loser.elixir.average", y="winner.elixir.average",
    alpha=0.4, s=20
)
x = np.linspace(2,7,100)
plt.plot(x,x,"k--", lw=1)
plt.xlabel("Opponent Deck Elixir Cost")
plt.ylabel("Winner Deck Elixir Cost")
plt.title("Elixir Matchups: Winners vs Opponents")
plt.tight_layout()
plt.savefig("elixir_matchup_scatter.png", dpi=300)
plt.close()

top_ranges = df["trophy_range"].value_counts().index.tolist()
plt.figure(figsize=(9,5))
sns.violinplot(
    data=df[df["trophy_range"].isin(top_ranges)],
    x="trophy_range", y="winner.elixir.average", inner="quartile"
)
plt.xlabel("Trophy Range")
plt.ylabel("Winner Average Elixir")
plt.title("Winner Elixir by Trophy Range")
plt.tight_layout()
plt.savefig("winner_elixir_by_trophy_range_violin.png", dpi=300)
plt.close()

edges = np.arange(2.0, 6.6, 0.2)
labels_bins = [f"{edges[i]:.1f}-{edges[i+1]:.1f}" for i in range(len(edges)-1)]

def bin_rate(sub):
    counts = pd.cut(sub["winner.elixir.average"], bins=edges, labels=labels_bins, include_lowest=True).value_counts().sort_index()
    total = sub.shape[0]
    return (counts/total).rename("win_rate_proxy")

winrate_before = bin_rate(df[df["patch_period"]=="Before"])
winrate_after  = bin_rate(df[df["patch_period"]=="After"])
winrate_df = pd.concat([winrate_before, winrate_after], axis=1)
winrate_df.columns = ["Before","After"]
winrate_df.to_csv("winrate_by_elixir_bins_before_after.csv")

plt.figure(figsize=(10,5))
plt.plot(winrate_df.index, winrate_df["Before"], marker="o", label="Before")
plt.plot(winrate_df.index, winrate_df["After"], marker="o", label="After")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Share of Matches in Bin (proxy for win density)")
plt.title("Elixir Bins — Before vs After Patch")
plt.legend()
plt.tight_layout()
plt.savefig("winrate_by_elixir_bin_before_after.png", dpi=300)
plt.close()
