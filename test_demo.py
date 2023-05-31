import numpy as np
import torch
# a = [1, 2]
# b = [3, 4]
# c = [(d, e) for d, e in zip(a, b)]
# print(c)
# for i, j in c:
#     print(i)
#     print(j)
#     print('------')
targets = [{'boxes': torch.tensor([[102.2756,  81.9204, 380.1563, 375.0000],
                                   [126.3799,  30.4097, 310.4151, 317.9615],
                                   [ 36.7432,  60.1282, 371.5597, 263.0226],
        [128.3867,  73.9969, 236.5653, 304.4479],
        [218.8822, 158.0554, 371.7212, 375.0000],
        [138.4869, 156.1199, 445.7796, 339.7021],
        [263.9427, 120.6564, 347.9532, 356.2540]]), 'labels': torch.tensor([15, 15, 15, 15, 15, 15, 15]),
            'scores': torch.tensor([0.3092, 0.1926, 0.1508, 0.1175, 0.0906, 0.0597, 0.0532])}]
a = []
for box in targets[0]['boxes']:
    i, j, k, l = box
    l = l.item()
    a.append(l)
print(np.mean(a))

# a = torch.tensor([1])
# b = torch.tensor([2])
# if a<0:
#     print('ok')
