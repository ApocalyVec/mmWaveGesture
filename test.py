from utils import label

# df, not_found = label('/Users/hanfei/label_2/label_2', '/Users/hanfei/onNotOn_data.csv')
#
# print(len(not_found))
# not_found = sorted(not_found)
# with open('missing_timestamp.txt', 'w') as writer:
#     for ts in not_found:
#         writer.write(str(ts) + '\n')
# writer.close()
# print('finished')
import os
lst = []
# path = '/Users/hanfei/onNoton_figures'
path = '/Users/hanfei/figures'
data_file = '/Users/hanfei/onNotOn_data.csv'
for folder in os.listdir(path):
    if not (folder == '.DS_Store' or folder == 'utils'):
        print(folder)
        sub_list = os.listdir(os.path.join(path, folder))
        if '.DS_Store' in sub_list: sub_list.remove('.DS_Store')
        sub_list = list(map(lambda x: x.strip('.jpg')[x.index('_')+1:], sub_list))
        lst.extend(sub_list)

assert isinstance(lst[0], str)

import pandas as pd
data = pd.read_csv(data_file)

timestamp_set = set()
for timestamp in range(len(data)):
    timestamp_set.add(str(data.loc[timestamp].iat[2]) + '_' + str(data.loc[timestamp].iat[3]))

assert isinstance(timestamp_set[0], str)

print('# of timestamps from images: ' + str(len(lst)))
print('# of timestamps from csv: ' + str(len(timestamp_set)))

result = []
for item in lst:
    if item not in timestamp_set: result.append(item)

print('# of missing timestamps: ' + str(len(result)))
print('example: ' + result[0])
