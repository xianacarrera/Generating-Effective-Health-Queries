import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

plot_type = "BM25 sobre diferentes corpus"
data = pd.read_csv('beir_corpus_stats.csv')

# time_taken is in the format "0:00:00.000000" (hours:minutes:seconds.microseconds)
# We convert it to minutes
data["time_taken_minutes"] = pd.to_timedelta(data["time_taken"]).dt.total_seconds() / 60
print(data.head())

# Add a column that instead of the full model name, only has a number
models = list(data["full_model"].unique())

# Metrics to plot
metrics = ["MAP@10", "MAP@100", "MAP@1000", "P@10", "P@100", "P@1000", "Recall@10", "Recall@100", "Recall@1000", "NDCG@10", "NDCG@100", "NDCG@1000"]


#############################################################
###### Bar plots
#############################################################


# Create a bar plot for each metric
fig, axes = plt.subplots(4, 3, figsize=(20, 30))
fig.suptitle(f"Comparación entre {plot_type}", fontsize=20)

# Different colors for each model
colors = sns.color_palette("husl", 4)

for i, metric in enumerate(metrics):
    ax = axes[i // 3, i % 3]
    # Create a barplot. Choose a different color for each axis row
    sns.barplot(data=data, x="abbreviation", y=metric, ax=ax, color=colors[i // 3],
                zorder = 3)
    ax.set_title(metric) 
    ax.set_xlabel("")
    # Rotate the x-axis labels and make them smaller
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=12)

    # Grid with transparency alpha=0.3 and behind the bars
    ax.grid(alpha=0.3, zorder = 0)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.8)
plt.show()




#############################################################
###### Heat map
#############################################################


# Create 4 heatmaps, one for each type of metric (MAP, P, Recall, NDCG)
fig, axes = plt.subplots(2, 2, figsize=(20, 20))

fig.suptitle(f"Comparación entre {plot_type}", fontsize=20)


for i, metric_type in enumerate(["MAP", "P", "Recall", "NDCG"]):
    metric_data = data[["abbreviation", f"{metric_type}@10", f"{metric_type}@100", f"{metric_type}@1000"]]
    heatmap_data = metric_data.set_index("abbreviation")
    ax = axes[i // 2, i % 2]

    sns.heatmap(heatmap_data, annot=True, cmap="YlGnBu", fmt=".3f", ax=ax)
    ax.set_title(metric_type)
    ax.set_xlabel("Métricas")
    ax.set_ylabel("Modelo")

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.1, hspace=0.3)
plt.show()


#############################################################
###### Time taken
#############################################################

# Create a bar plot for the time taken
fig, ax = plt.subplots(figsize=(20, 12))

# Make the plot ordered by time taken
data_sorted = data.sort_values("time_taken_minutes", ascending=False)
sns.barplot(data=data_sorted, x="abbreviation", y="time_taken_minutes", ax=ax, zorder = 3)
ax.set_title(f"Tiempo de ejecución ({plot_type})")
ax.set_xlabel("Modelo")
ax.set_ylabel("Tiempo (minutos)")
# Rotate the x-axis labels and make them smaller
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, fontsize=8)

# Grid with transparency alpha=0.3 and behind the bars
plt.grid(alpha=0.3, zorder = 0)

plt.tight_layout()
plt.subplots_adjust(top=0.92, bottom=0.2)
plt.show()



#############################################################
###### Statistics
#############################################################

# Find the best model for each metric
for metric in metrics:
    best_model = data.loc[data[metric].idxmax()]["abbreviation"]
    print(f"Best model for {metric}: {best_model} ---> {data[metric].max()}")

