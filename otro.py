import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'data_extraction_SSL_on_drones.csv'  # Replace with the correct path if necessary
data = pd.read_csv(file_path)

# Filter rows where 'Mics' is not numeric
data = data[pd.to_numeric(data['Mics'], errors='coerce').notna()]  # Only numeric values in 'Mics'

# Group data to count each combination of Mics and DOA execution location
grouped_data = data.groupby(['Mics', 'Where is the DOA algorithm executed? ']).size().unstack(fill_value=0)

# Create stacked bar plot
fig, ax = plt.subplots(figsize=(12, 8))
grouped_data.plot(kind='bar', stacked=True, ax=ax, colormap='viridis')

# Plot configurations
plt.xlabel("Number of microphones in the array", fontsize=20)
plt.ylabel("Articles reviewed", fontsize=20)
plt.yticks(ticks=range(0, 20, 2), fontsize=20)  # Multiples of two for y-axis
plt.xticks(ticks=range(len(grouped_data.index)), labels=grouped_data.index.astype(int), rotation=30, ha='right', fontsize=20)  # X-axis labels as integers without floating points
plt.legend(title="DOA Execution", loc='best', fontsize=16)
plt.tight_layout()
plt.show()

# Filter rows where 'Mics' is not numeric and exclude specific configurations
data = data[pd.to_numeric(data['Mics'], errors='coerce').notna()]  # Only numeric values in 'Mics'

# Group data to count each combination of Mics and configuration
grouped_data_config = data.groupby(['Mics', 'configuration']).size().unstack(fill_value=0)

# Create stacked bar plot
fig, ax = plt.subplots(figsize=(12, 8))
grouped_data_config.plot(kind='bar', stacked=True, ax=ax, colormap='jet')

# Plot configuration
plt.xlabel("Number of microphones in the array", fontsize=20)
plt.ylabel("Articles reviewed", fontsize=20)
plt.yticks(ticks=range(0, 20, 2), fontsize=20)  # Multiples of two for y-axis
plt.xticks(ticks=range(len(grouped_data_config.index)), labels=grouped_data_config.index.astype(int), rotation=30, ha='right', fontsize=20)  # X-axis labels as integers without floating points
plt.legend(title="Configuration", loc='best', fontsize=16)
plt.tight_layout()
plt.show()

config_counts = data['configuration'].value_counts()

# Create bar plot for configuration counts
fig, ax = plt.subplots(figsize=(12, 8))
config_counts.plot(kind='bar', ax=ax)

# Plot configuration
plt.xlabel("Configuration", fontsize=20)
plt.ylabel("Articles reviewed", fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()

sampling_counts = data['samp'].value_counts()

fig, ax = plt.subplots(figsize=(12, 8))
sampling_counts.plot(kind='bar', ax=ax)

    # Plot configuration
plt.xlabel("Sampling Rate in KHz", fontsize=23)
plt.ylabel("Articles reviewed", fontsize=23)
plt.xticks(rotation=45, ha='right', fontsize=25)
plt.yticks(fontsize=25)
plt.tight_layout()
plt.show()



# Filter rows where 'Mics' is not numeric
data = data[pd.to_numeric(data['samp'], errors='coerce').notna()]  # Only numeric values in 'Mics'

# Group data to count each combination of Mics and DOA execution location
grouped_data = data.groupby(['samp', 'sound']).size().unstack(fill_value=0)

# Create stacked bar plot
fig, ax = plt.subplots(figsize=(10, 9))
grouped_data.plot(kind='bar', stacked=True, ax=ax, colormap='jet')

# Plot configurations
plt.xlabel("Sampling Rate", fontsize=20)
plt.ylabel("Articles reviewed", fontsize=20)
plt.yticks(ticks=range(0, 20, 2), fontsize=20)  # Multiples of two for y-axis
plt.xticks(ticks=range(len(grouped_data.index)), labels=grouped_data.index.astype(int), rotation=30, ha='right', fontsize=20)  # X-axis labels as integers without floating points
plt.legend(loc='best', fontsize=16)
plt.tight_layout()
plt.show()