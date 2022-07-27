import numpy as np
from sklearn.preprocessing import scale
# a=np.array([1,0,1,0,1,0,1,1,0,1]) #real
# b=np.array([1,1,0,0,0,0,1,1,0,1]) #pred
# c=a-b
# error_indices=c.nonzero()
# print(error_indices[0])
# errors=len(error_indices[0])
# true_positives= 78
# true_negatives= 3
# false_positives= 10
# false_negatives=3
# # true_positives= len(set(np.where(a == 1)[0]).intersection(set(np.where(b == 1)[0])))
# # true_negatives= len(set(np.where(a == 0)[0]).intersection(set(np.where(b == 0)[0])))
# # false_positives=len(set(np.where(a == 0)[0]).intersection(set(np.where(b == 1)[0])))
# # false_negatives=len(set(np.where(a == 1)[0]).intersection(set(np.where(b == 0)[0])))
#
# precision=true_positives/(true_positives+false_positives)
# recall=true_positives/(true_positives+false_negatives)
# f1=2*precision*recall/(precision+recall)
#
# precision_neg=true_negatives/(true_negatives+false_negatives)
# recall_neg=true_negatives/(true_negatives+false_positives)
# f1_neg=2*precision_neg*recall_neg/(precision_neg+recall_neg)
#
#
# accuracy=(true_positives+true_negatives)/(true_positives+true_negatives+false_positives+false_negatives)
#
# # print(errors)
# print("true_positives "+str(true_positives))
# print("true_negatives "+str(true_negatives))
# print("false_positives "+str(false_positives))
# print("false_negatives "+str(false_negatives))
# print("precision "+str(precision))
# print("recall "+str(recall))
# print("f1 "+str(f1))
# print("precision_neg "+str(precision_neg))
# print("recall_neg "+str(recall_neg))
# print("f1_neg "+str(f1_neg))
# print((f1+f1_neg)/2)
# print("accuracy "+str(accuracy))


b=np.array([[7,1,100,10],
           [5,0,50000,1],
           [3,-1,1500,0]])
a=np.array([3,2,1,0,2,3,2,8,1])
# np.array(a[2:], a[:2])
# print(str(np.column_stack((a[2:], a[:2]))))
b=np.array([a[2:], a[1:-1], a[:-2]])
print(str(np.array([a[2:], a[1:-1], a[:-2]]).shape))
class_array = []
for i in range(b.shape[1]):
    x = b[:,i]
    x=x[::-1]
    print("xxx "+str(x))
    print("x[0] < x[1] "+ str(x[0] < x[1]))
    print("x[1] > x[2] "+ str(x[1] < x[2]))
    if x[0] > x[1] and x[1] > x[2]:   # 3 2 1
        class_array.append(1) # =
    elif x[0] > x[1] and x[1] < x[0]:  # 3 2 3
        class_array.append(2) # sube
    elif x[0] > x[1] and x[1] == x[2]: # 3 2 2
        class_array.append(1)  # =
    elif x[0] == x[1] and x[1] > x[2]: # 3 3 2
        class_array.append(0)  # baja
    elif x[0] == x[1] and x[1] < x[2]: # 3 3 4
        class_array.append(2)  # sube
    elif x[0] == x[1] and x[1] == x[2]: # 3 3 3
        class_array.append(1)  # =
    elif x[0] < x[1] and x[1] > x[2]:   # 1 2 1
        class_array.append(0) # baja
    elif x[0] < x[1] and x[1] < x[2]:  # 1 2 3
        class_array.append(1) # =
    elif x[0] < x[1] and x[1] == x[2]: # 1 2 2
        class_array.append(1)  # =
    print(class_array)
np.concatenate([[5],np.array(class_array )])
print(str(np.concatenate([[5],np.array(class_array )])))

np.insert(np.array(class_array ),0, 5)
print(str(np.insert(np.array(class_array ),0, 5)))



