import numpy as np
# import matplotlib as mpl
#
# ## agg backend is used to create plot as a .png file
# mpl.use('agg')

import matplotlib.pyplot as plt


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
# print(os.getcwd())
df = pd.read_csv('./EJPH/violin.csv')

sns.set(style="whitegrid", palette="pastel", color_codes=True)

f, ax = plt.subplots(figsize=(8, 8))

# Draw a nested violinplot and split the violins for easier comparison
sns.violinplot(x="feed", y="weight", hue="sex", data=df, split=True, linewidth=2.5,
               inner="quart", palette={"male": "b", "female": "y"})
sns.despine(left=True)



f.suptitle('Chick weights by feed type', fontsize=18, fontweight='bold')
ax.set_xlabel("Feed",size = 16,alpha=0.7)
ax.set_ylabel("Weight (g)",size = 16,alpha=0.7)
plt.legend(loc='upper left')
plt.savefig('./EJPH/violin.pdf')
plt.show()

# N_TRIALS=10
# THETA_DOT_THRESHOLD=10
# theta=np.linspace(-np.pi,np.pi,N_TRIALS)
# theta_dot=np.linspace(-THETA_DOT_THRESHOLD,THETA_DOT_THRESHOLD,N_TRIALS)
# arr=np.transpose([np.tile(theta, len(theta_dot)), np.repeat(theta_dot, len(theta))])
# plt.plot(arr[:,0],arr[:,1],'.')
# plt.savefig('./EJPH/plots/iniSpace.pdf')
# plt.show()
