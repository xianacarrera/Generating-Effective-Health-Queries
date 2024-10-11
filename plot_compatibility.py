import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the compatibility results
data = pd.read_csv('compatibility.csv')

"""
helpful_data = data[data["qrels"] == "misinfo-qrels-graded.helpful-only"]
print(helpful_data.head()) 

harmful_data = data[data["qrels"] == "misinfo-qrels-graded.harmful-only"]
print(harmful_data.head())
"""

# Delete all file extensions from the run column
data["run"] = data["run"].str.replace(".txt", "").str.replace(".csv", "")
print(data.head())

# Map the original qrels labels to 'helpful' and 'harmful'
label_mapping = {
    "misinfo-qrels-graded.helpful-only": "helpful",
    "misinfo-qrels-graded.harmful-only": "harmful"
}
data["qrels"] = data["qrels"].map(label_mapping)


sns.set_theme(style="whitegrid")

g = sns.catplot(x = 'run', y='all', 
            hue = 'qrels', data=data,
            kind='bar', height=6, aspect=2, palette={"helpful": "#5DADEB", "harmful": "#8C0000"})
# Helpful is blue, harmful is red

# Rotate the x-axis labels and make them smaller
g.set_xticklabels(rotation=45, fontsize=8)
g.set_xlabels("Modelo")
g.set_ylabels("Compatibilidad media")
g.figure.suptitle("Comparativa de compatibilidad entre modelos", fontsize=15)

# Move the legend to the upper right corner
legend = g._legend
legend.set_title("Qrels")
legend.get_frame().set_edgecolor('black')   # Add a black border
legend.get_frame().set_linewidth(1)         # Set the border width

sns.move_legend(g, "upper right", frameon=True, facecolor="lightgrey")

# Add a grid
g.ax.grid(True, which='major', axis='y', linestyle='--')

plt.tight_layout()
plt.show()


#####################################################
# Compatibility by topic
#####################################################



cols = [str(i) for i in range(1, 51)]

melted = pd.melt(data, id_vars=['run', 'qrels'], value_vars=cols, var_name='x', value_name='compatibilidad')

plt.figure(figsize=(25, 12))
sns.set_palette("husl")

ax = sns.boxplot(x='x', y='compatibilidad', hue='qrels', data=melted)

plt.xlabel("Topic")
plt.ylabel("Compatibilidad")
plt.title("Compatibilidad por topic")

# Increase x-tick separation
plt.setp(ax.get_xticklabels(), ha='right')
ax.set_xticks(range(len(ax.get_xticklabels())))
ax.set_xticklabels(ax.get_xticklabels(), ha='right')

plt.tight_layout()
plt.show()


#####################################################
# Helpful - harmful 
#####################################################

# Compute a new dataframe with the difference between helpful and harmful for each model
helpful_data = data[data["qrels"] == "helpful"]
harmful_data = data[data["qrels"] == "harmful"]

# Merge the two dataframes
diff_data = pd.merge(helpful_data, harmful_data, on="run", suffixes=('_helpful', '_harmful'))

print(diff_data.head())

# Compute the difference between the two compatibilities
diff_data["all"] = diff_data["all_helpful"] - diff_data["all_harmful"]
diff_data["all-util"] = 2 * diff_data["all_helpful"] - diff_data["all_harmful"]

for i in range(1, 51):
    diff_data[str(i)] = diff_data[str(i) + "_helpful"] - diff_data[str(i) + "_harmful"]
    diff_data[str(i) + "-util"] = 2 * diff_data[str(i) + "_helpful"] - diff_data[str(i) + "_harmful"]

# Remove the columns that are not needed
diff_data = diff_data[["run", "all", "all-util"] + [str(i) for i in range(1, 51)] + [str(i) + "-util" for i in range(1, 51)]]
print(diff_data.head())

# Plot the difference between helpful and harmful
sns.set_theme(style="whitegrid")

g = sns.catplot(x = 'run', y='all', 
            data=diff_data.sort_values(by="all", ascending=False), kind='bar', height=6, aspect=2)

# Rotate the x-axis labels and make them smaller
g.set_xticklabels(rotation=45, fontsize=8)
g.set_xlabels("Modelo")
g.set_ylabels("Helpful - Harmful")
g.figure.suptitle("Comparativa de helpful - harmful entre modelos", fontsize=15)

# Add a grid
g.ax.grid(True, which='major', axis='y', linestyle='--')

plt.tight_layout()
plt.show()

# Plot the difference between 2 * helpful and harmful
sns.set_theme(style="whitegrid")

g = sns.catplot(x = 'run', y='all-util', 
            data=diff_data.sort_values(by="all-util", ascending=False), kind='bar', height=6, aspect=2)

# Rotate the x-axis labels and make them smaller
g.set_xticklabels(rotation=45, fontsize=8)
g.set_xlabels("Modelo")
g.set_ylabels("2 * Helpful - Harmful")
g.figure.suptitle("Comparativa de  2 *helpful - harmful entre modelos", fontsize=15)

# Add a grid
g.ax.grid(True, which='major', axis='y', linestyle='--')

plt.tight_layout()
plt.show()
