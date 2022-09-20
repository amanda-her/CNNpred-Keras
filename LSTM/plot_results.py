import pandas as pd
import matplotlib.pyplot as plt

# model="LSTM"
# inputs=[
#     [[60,1,200,128,True],"LSTM-1-all"],
#     [[60,4,200,128,True],"LSTM-4-all"],
#     [[60,83,200,128,True],"LSTM-83-all"],
#     [[60,1,200,128,False],"LSTM-1-single"],
#     [[60,4,200,128,False],"LSTM-4-single"],
#     [[60,83,200,128,False],"LSTM-83-single"]
# ]
# model="CNN"
# inputs=[
#     [[60,1,200,128,True],"{}-1-all".format(model)],
#     [[60,4,200,128,True],"{}-4-all".format(model)],
#     [[60,83,200,128,True],"{}-83-all".format(model)],
#     [[60,1,200,128,False],"{}-1-single".format(model)],
#     [[60,4,200,128,False],"{}-4-single".format(model)],
#     [[60,83,200,128,False],"{}-83-single".format(model)]
# ]
model="CNN-LSTM"
inputs=[
    [[60,1,200,128,True],"{}-1-all".format(model)],
    [[60,4,200,128,True],"{}-4-all".format(model)],
    [[60,83,200,128,True],"{}-83-all".format(model)],
    [[60,1,200,128,False],"{}-1-single".format(model)],
    [[60,4,200,128,False],"{}-4-single".format(model)],
    [[60,83,200,128,False],"{}-83-single".format(model)]
]
results =[]
fig, ax = plt.subplots()
ax.grid(True, axis='y', zorder=0)
for input in inputs:
    def build_name(a,b,c,d,e):
        # return "results-{}-{}-{}-{}-{}-64-5.csv".format(a,b,c,d,e)
        # return "results-CNNpred-{}-{}-{}-{}-{}-8-0-5.csv".format(a,b,c,d,e)
        return "results-version2-CNN-LSTM-{}-{}-{}-{}-{}-8-64-5.csv".format(a,b,c,d,e)
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
plt.savefig("{}-version2-summary.jpg".format(model))

