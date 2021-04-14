import re
import numpy as np

output_old = open('ndcg-old.txt', 'r')
output_new = open('ndcg-new.txt', 'r')

array = []
for line in output_old:
    values_old = re.split('\s+', line.strip())
    values_new = re.split('\s+', output_new.readline().strip())
    values_old.append(float(values_new[2]) - float(values_old[2]))
    array.append(values_old)

array = np.array(array)
array = array[array[:, 3].argsort()]
print(array)