import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import MDS
import seaborn as sns
import matplotlib.cm as cm

df = pd.read_csv('/Users/boyangzhang/Downloads/Research Methods/course work2/Results_21Mar2022.csv')
print(df.head())
#Keep the columns you want
columns_to_keep = ['diet_group', 'sex', 'age_group','mean_ghgs','mean_land','mean_watscar','mean_acid']
df_filtered = df[columns_to_keep]

#Save as a new CSV file
df_filtered.to_csv('/Users/boyangzhang/Downloads/Research Methods/course work2/Data processing/my_filtered_output.csv', index=False)
print(df_filtered.head())

#Check the number of missing values for each column
df = pd.read_csv("/Users/boyangzhang/Downloads/Research Methods/course work2/Data processing/my_filtered_output.csv")
missing_info = df_filtered.isnull().sum()
print("\nThe number of missing values per column (attribute).：")
print(missing_info)
df = pd.read_csv("/Users/boyangzhang/Downloads/Research Methods/course work2/Data processing/my_filtered_output.csv")

# Detect and remove outliers using IQR
for col in ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_acid']:
    Q1 = df_filtered[col].quantile(0.25)
    Q3 = df_filtered[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outlier_count = df_filtered[(df_filtered[col] < lower_bound) | (df_filtered[col] > upper_bound)].shape[0]
    print(f"{col} - Outliers removed: {outlier_count}")
    df_filtered = df_filtered[(df_filtered[col] >= lower_bound) & (df_filtered[col] <= upper_bound)]

# pandas
grouped_df = df_filtered.groupby(['diet_group', 'sex'], as_index=False).agg({
    'mean_ghgs': 'mean',
    'mean_land': 'mean',
    'mean_watscar': 'mean',
    'mean_acid': 'mean'
})
# Normalization
scaler = MinMaxScaler()
# Normalize the environmental indicators of grouped_df
scaler = MinMaxScaler()
cols_to_normalize = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_acid']

# Overwrite the original column directly
grouped_df[cols_to_normalize] = scaler.fit_transform(grouped_df[cols_to_normalize])

# Merge the normalized results back into the original dataframe
male_df = grouped_df[grouped_df['sex'] == 'male'].set_index('diet_group')
female_df = grouped_df[grouped_df['sex'] == 'female'].set_index('diet_group')

# Whether there is the same diet_group, do inner join or reflow
common_diets = male_df.index.intersection(female_df.index)
male_df = male_df.loc[common_diets]
female_df = female_df.loc[common_diets]

# Radar chart
labels = ['mean_ghgs', 'mean_land', 'mean_watscar', 'mean_acid']
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

color_map = cm.get_cmap('tab20', len(common_diets))
fig, ax = plt.subplots(figsize=(9.5, 7), subplot_kw=dict(polar=True))
for i, diet in enumerate(common_diets):
    male_values = male_df.loc[diet, labels].tolist() + [male_df.loc[diet, labels[0]]]
    female_values = female_df.loc[diet, labels].tolist() + [female_df.loc[diet, labels[0]]]
    color = color_map(i)

    # Men: solid line + dot
    ax.plot(angles, male_values, label=f'{diet.capitalize()} - Male',
            linestyle='solid', linewidth=1, marker='o', color=color)

    # Women: Dotted line + square
    ax.plot(angles, female_values, label=f'{diet.capitalize()} - Female',
            linestyle='dashed', linewidth=1, marker='s', color=color)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), ['GHG', 'Land', 'WaterScarcity', 'Acidification'])

ax.set_title('Environmental Impact by Diet and Gender', size=18, pad=20)
ax.legend(loc='center left', bbox_to_anchor=(1.15, 0.9), title="Groups", fontsize=9)

plt.tight_layout()
plt.show()
# Heatmap
diff_df = male_df.copy()
for col in labels:
    diff_df[col] = male_df[col] - female_df[col]

diff_df.rename(columns={
    'mean_ghgs': 'GHG',
    'mean_land': 'Land',
    'mean_watscar': 'WaterScarcity',
    'mean_acid': 'Acidification'
}, inplace=True)

# build Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 4))
sns.heatmap(
    diff_df[['GHG', 'Land', 'WaterScarcity', 'Acidification']],
    cmap='coolwarm',
    center=0,
    annot=True,
    linewidths=0.5,
    fmt=".2f",
    cbar_kws={'label': 'Male - Female Difference'}
)
plt.title("Gender-Based Environmental Impact Differences (Male - Female)", fontsize=13)
plt.ylabel("Diet Group")
plt.xlabel("Environmental Indicators")
plt.tight_layout()
plt.show()