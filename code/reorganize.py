import pandas as pd
import glob
import subprocess

a = 'a3'
shadow_df = pd.read_csv(f'models/{a}/{a}_shadow_attr.csv', index_col=0)
# test_df = pd.read_csv(f'models/{a}/{a}_test_attr.csv', index_col=0)
# valid_df = pd.read_csv(f'models/{a}/{a}_valid_attr.csv', index_col=0)



row_list = []
for i, file in enumerate(sorted(glob.glob(f'models/{a}/*.pth'))):
    print(file)
    filename = file.split('/')[-1]
    print(filename)
    split = filename.split('_')[1]
    print(split)
    architecture = filename.split('_')[0]
    print(architecture)

    if split == 'shadow':
        print('shadow', filename[3:-4])
        # dist = shadow_df[shadow_df['model'] == f'{filename[3:-4]}']['male_dist'].values[0]
        dist = shadow_df[shadow_df['model'] == f'{a}_{filename[3:-4]}']['male_dist'].values[0]
        split_val = 0
        pass
    elif split == 'test':
        print('test', filename[3:-4])
        # dist = test_df[test_df['model'] == f'{filename[3:-4]}']['male_dist'].values[0]
        dist = test_df[test_df['model'] == f'{a}_{filename[3:-4]}']['male_dist'].values[0]
        split_val = 1
        pass
    elif split == 'valid':
        print('valid', filename[3:-4])
        # dist = valid_df[valid_df['model'] == f'{filename[3:-4]}']['male_dist'].values[0]
        dist = valid_df[valid_df['model'] == f'{a}_{filename[3:-4]}']['male_dist'].values[0]
        split_val = 2
        pass
    else:
        print('wrong split:', split)
    subprocess.run(['cp', file, f'models/{i+3600}.pth'])

    row_list.append({
        'model':f'{i+3600}.pth',
        'male_dist':dist,
        'split':split_val,
        'architecture':architecture
    })
    # break
    # if i > 10: break

new_df = pd.DataFrame(row_list)
new_df = new_df.set_index(pd.Index(range(3600, 3600+len(row_list))))
print(new_df)
new_df.to_csv('models/models.csv', mode='a', header=False)
