import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from functools import reduce

df1 = pd.read_csv("vinfo_data/attn_ptbpos_values.csv")
df2 = pd.read_csv("vinfo_data/attnres_ptbpos_values.csv")
df3 = pd.read_csv("vinfo_data/attnresln_ptbpos_values.csv")

df1["Type"] = "ATTN"
df2["Type"] = "ATTNRES"
df3["Type"] = "ATTNRESLN"

df = pd.concat([df1, df2, df3], ignore_index = True)

plt.figure(figsize=(10, 6))
sb.lineplot(data=df, x='Layer', y='Conditional', hue='Type')
plt.title('V-Information (Conditional) (PTB_POS)')
plt.xlabel('Layer')
plt.ylabel('V Entropy (Conditional)')
plt.legend(title='Type')
#plt.grid(True)

output_file = 'vinfo_plots/v_info_ptbpos_conditional_plot.png'
plt.savefig(output_file)