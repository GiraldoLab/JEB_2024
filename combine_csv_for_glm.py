import pandas as pd

# Define the names of your CSV files
file_names = ['20min_4suns_glm_data.csv',   'diff_sun_continuous_glm_data.csv', 'diff_sun_interrupted_glm_data.csv',  '3suns_PB.csv' ]

# Read each file into a DataFrame and store all DataFrames in a list
dfs = [pd.read_csv(file_name) for file_name in file_names]

# Concatenate all DataFrames together
combined_df = pd.concat(dfs, ignore_index=True)

# Write the combined DataFrame to a new CSV file
combined_df.to_csv('fig2_diff_sun_data.csv', index=False)
