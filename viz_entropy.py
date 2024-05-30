import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

df = pd.read_csv('attnresln_values.csv')

# Melt the DataFrame to have a long-form structure suitable for seaborn
df_melted = df.melt(id_vars=['Layer'], value_vars=['Baseline', 'Conditional'],
                    var_name='Type', value_name='Entropy')

df_melted['Entropy'] = pd.to_numeric(df_melted['Entropy'], errors='coerce')

plt.figure(figsize=(10, 6))
sb.lineplot(data=df_melted, x='Layer', y='Entropy', hue='Type')
plt.title('V-Information for ATTNRESLN')
plt.xlabel('Layer')
plt.ylabel('Entropy')
plt.legend(title='Type')
#plt.grid(True)

output_file = 'attnresln_plot.png'
plt.savefig(output_file)