import pandas as pd
import matplotlib.pyplot as plt

inputs=[
    [[60,1,200,128,True],"1 feature, all stocks"],
    [[60,4,200,128,True],"4 features, all stocks"],
    [[60,1,500,128,False],"1 feature, single stock"],
    [[60,4,500,128,False],"4 features, single stock"]
]
results =[]
fig, ax = plt.subplots()
ax.grid(True, axis='y', zorder=0)
for input in inputs:
    def build_name(a,b,c,d,e):
        return "results-{}-{}-{}-{}-{}.csv".format(a,b,c,d,e)
    data1 = pd.read_csv(build_name(*(input[0])), header=None).to_numpy()
    data1=data1[1:]
    mae = data1[:,1].astype("float")
    print(mae)
    labels=data1[:,0]
    xticks=range(0, len(mae))
    ax.plot(xticks,mae,marker='o', zorder=10, label=input[1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(labels)
    results.append(mae)

ax.legend()
plt.savefig("summary.jpg")

