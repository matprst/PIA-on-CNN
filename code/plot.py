import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# df = pd.read_csv('./models/attack_models/attack_models_test.csv', index_col=0)
# architectures = sorted(list(set(df[['architecture', 'num_param']].itertuples(index=False))), key=lambda x: x.num_param)
# print(architectures)
# accuracies = []
# weights = []
# for a in architectures:
#     acc = df[df['architecture'] == a.architecture]['accuracy']
#     print(acc)
#     accuracies.append(np.mean(acc.values) * 100)
#     weights.append(a.architecture)
# plt.bar(weights, accuracies)
# plt.ylim(0, 100)
# plt.xlabel('Architecture (from fewer to larger #params)')
# plt.ylabel('Accuracy (%)')
# props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# # plt.text(3.9, 95, 'a1: 2 convs - 3 fcs\na2: 1 convs - 3 fcs\na3: 3 convs - 3 fcs\na4: 3 convs - 2 fcs\na5: 3 convs - 1 fcs\na6: 2 convs - 2 fcs', fontsize=10,
# #         verticalalignment='top', bbox=props)
# plt.title('Attack accuracies')
# plt.show()

def plot_accuracies_shadows():
    df = pd.read_csv('./models/models_perf.csv', index_col=0)
    # df['attack'] = df['model'].str.split('-').str[1]
    df['accuracy'] = df['accuracy'] * 100
    df['precision'] = df['precision'] * 100
    df['recall'] = df['recall'] * 100
    print(df)
    groups = df.groupby(['architecture']).mean()
    # groups = groups.sort_values('num_param')
    # print(plt.plot(groups.loc['1'].accuracy))
    print(groups)
    print(groups.accuracy)
    ax = groups.plot(kind='bar')

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylabel('accuracy (%)')
    plt.xlabel('shadow model architecture')
    plt.title('Perf of shadow models')
    plt.ylim(0, 100)
    plt.show()


def plot_accuracies_per_architecture_full():
    df = pd.read_csv('./models/attack_models/full/attack_models.csv', index_col=0)
    print(df)
    df['precision'] = df['precision'].replace(['//'], 1.0)
    df['precision'] = pd.to_numeric(df['precision'])
    print(df)
    # print(df.groupby(['architecture']).median())
    df['attack'] = df['model'].str.split('-').str[1]
    # df['accuracy'] = df['accuracy'] * 100
    print(df)
    # groups = df[df['attack'] == '1'].groupby(['architecture']).mean()
    # print(df)
    # print(df[df['attack'] == '1'].groupby(['architecture']).median())
    final_df = pd.DataFrame({
        'accuracy': df[df['attack'] == '1'].groupby(['architecture']).median().accuracy * 100 ,
        'precision': df[df['attack'] == '1'].groupby(['architecture']).median().precision * 100 ,
        'recall': df[df['attack'] == '1'].groupby(['architecture']).median().recall * 100 ,
        'accuracy_std': df[df['attack'] == '1'].groupby(['architecture']).std().accuracy * 100 ,
        'precision_std': df[df['attack'] == '1'].groupby(['architecture']).std().precision * 100 ,
        'recall_std': df[df['attack'] == '1'].groupby(['architecture']).std().recall * 100
    })
    print(final_df)

    # groups = groups.sort_values('num_param')
    # print(plt.plot(groups.loc['1'].accuracy))
    # print(groups)
    # print(groups.accuracy)
    # ax = groups.accuracy.plot(kind='bar', color='#1a9988')
    ax = final_df[['accuracy', 'precision', 'recall']].plot(kind='bar', color=['#1a9988', '#eb5600', '#6aa4c8'], yerr=final_df[['accuracy_std', 'precision_std', 'recall_std']].values.T, capsize=4, zorder=3)
    ax.grid(axis='y', zorder=0)

    # for p in ax.patches:
    #     ax.annotate(f'{int(p.get_height())}', (p.get_x() * 1.005, p.get_height() * 1.005))
    fontsize=20
    plt.ylabel('Metric (%)', fontsize=fontsize)
    plt.xlabel('shadow model architecture', fontsize=fontsize)
    plt.title('Attacks Performances', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.ylim(0, 100)
    ax.legend(fontsize=fontsize)
    plt.show()

def plot_accuracies_per_architecture_fcn():
    df = pd.read_csv('./models/attack_models_fcn/attack_models_fcn.csv', index_col=0)
    df['attack'] = df['model'].str.split('-').str[1]
    df['accuracy'] = df['accuracy'] * 100
    print(df)
    groups = df[df['attack'] == '1'].groupby(['architecture']).mean()
    # groups = groups.sort_values('num_param')
    # print(plt.plot(groups.loc['1'].accuracy))
    print(groups)
    print(groups.accuracy)
    ax = groups.accuracy.plot(kind='bar')

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylabel('accuracy (%)')
    plt.xlabel('shadow model architecture')
    plt.title('Only fcn')
    plt.ylim(0, 100)
    plt.show()

def plot_accuracies_per_architecture_conv():
    df = pd.read_csv('./models/attack_models_conv/attack_models_conv.csv', index_col=0)
    df['attack'] = df['model'].str.split('-').str[1]
    df['accuracy'] = df['accuracy'] * 100
    print(df)
    groups = df[df['attack'] == '1'].groupby(['architecture']).mean()
    # groups = groups.sort_values('num_param')
    # print(plt.plot(groups.loc['1'].accuracy))
    print(groups)
    print(groups.accuracy)
    ax = groups.accuracy.plot(kind='bar')

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylabel('accuracy (%)')
    plt.xlabel('shadow model architecture')
    plt.title('Only conv')
    plt.ylim(0, 100)
    plt.show()

def plot_accuracies_per_weights():
    df = pd.read_csv('./models/attack_models/full/attack_models.csv', index_col=0)
    df['attack'] = df['model'].str.split('-').str[1]
    df['accuracy'] = df['accuracy'] * 100
    print(df)
    groups = df[df['attack'] == '1'].groupby(['num_param']).median()
    groups = groups.sort_values('num_param')
    # print(plt.plot(groups.loc['1'].accuracy))
    print(groups)
    print(groups.accuracy)
    # ax = groups.accuracy.plot(kind='scatter', logx=True)

    df = groups.accuracy.to_frame()
    df.reset_index(inplace=True)
    df.columns = ['weights','accuracy']
    ax = df.plot(kind='scatter',x='weights',y='accuracy', logx=True, s=600, marker='.', c='#1a9988')
    ax.grid()

    z = df['weights'].values
    print(z)
    y = df['accuracy'].values
    print(y)
    fontsize = 20
    for i, txt in enumerate(y):
        ax.annotate(f'{txt:.1f}', (z[i]+1, y[i]+3), fontsize=fontsize)

    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylabel('accuracy (%)', fontsize=fontsize)
    plt.ylim(0, 100)
    plt.xlim(1000, 1000000)
    plt.xlabel('Number of weights in shadow model architecture', fontsize=fontsize)
    plt.title('Number of parameters influence', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)

    plt.show()

def plot_accuracies():
    df = pd.read_csv('./models/attack_models/full/attack_models.csv', index_col=0)
    df['attack'] = df['model'].str.split('-').str[1]
    df_fcn = pd.read_csv('./models/attack_models/fcn/attack_models_fcn.csv', index_col=0)
    df_conv = pd.read_csv('./models/attack_models/conv/attack_models_conv.csv', index_col=0)
    print(df[df['attack'] == '1'].groupby(['architecture']).mean().accuracy)
    print(df_fcn.groupby(['architecture']).mean().accuracy)
    print(df_conv.groupby(['architecture']).mean().accuracy)
    final_df = pd.DataFrame({
        'full': df[df['attack'] == '1'].groupby(['architecture']).median().accuracy * 100,
        'fcn': df_fcn.groupby(['architecture']).median().accuracy * 100,
        'conv': df_conv.groupby(['architecture']).median().accuracy * 100,
        'full_std': df[df['attack'] == '1'].groupby(['architecture']).std().accuracy * 100,
        'fcn_std': df_fcn.groupby(['architecture']).std().accuracy * 100,
        'conv_std': df_conv.groupby(['architecture']).std().accuracy * 100
    })
    # std_df = pd.DataFrame({
    #     'full': df.groupby(['architecture']).std().accuracy * 100,
    #     'fcn': df_fcn.groupby(['architecture']).std().accuracy * 100,
    #     'conv': df_conv.groupby(['architecture']).std().accuracy * 100
    # })
    print(final_df)
    # print(std_df)
    # groups = df[df['attack'] == '1'].groupby(['num_param']).mean()
    # groups = groups.sort_values('num_param')
    # print(plt.plot(groups.loc['1'].accuracy))
    # print(groups)
    # print(groups.accuracy)
    ax = final_df[['full', 'fcn', 'conv']].plot(kind='bar', color=['#1a9988', '#eb5600', '#6aa4c8'], yerr=final_df[['full_std', 'fcn_std', 'conv_std']].values.T, capsize=4, zorder=3)
    ax.grid(axis='y', zorder=0)
    # for p in ax.patches:
    #     ax.annotate(f'{int(p.get_height())}', (p.get_x() - .01, p.get_height() - .01), fontsize=15)
    fontsize = 20
    plt.ylabel('accuracy (%)', fontsize=fontsize)
    plt.xlabel('shadow model architecture', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    plt.title('Convolution vs FCN layers', fontsize=fontsize)
    plt.ylim(0, 100)
    plt.legend(fontsize=fontsize)
    plt.show()

def plot_accuracies_attacks():
    df = pd.read_csv('./models/attack_models/full/attack_models.csv', index_col=0)
    df['attack'] = df['model'].str.split('-').str[1]
    # print(df['model'].str.split('-').str[1])
    df = df[df['attack']<='3'][['attack', 'architecture', 'accuracy']]
    # print(df.groupby(['attack', 'architecture']).mean())
    # df_fcn = pd.read_csv('./models/attack_models_fcn/attack_models_fcn.csv', index_col=0)
    # df_conv = pd.read_csv('./models/attack_models_conv/attack_models_conv.csv', index_col=0)
    # print(df.groupby(['architecture']).mean().accuracy)
    # print(df_fcn.groupby(['architecture']).mean().accuracy)
    # print(df_conv.groupby(['architecture']).mean().accuracy)
    print(df.T)
    final_df = pd.DataFrame({
        '1': df[df['attack']=='1'].groupby(['architecture']).mean().accuracy * 100,
        '2': df[df['attack']=='2'].groupby(['architecture']).mean().accuracy * 100,
        '3': df[df['attack']=='3'].groupby(['architecture']).mean().accuracy * 100,
        '1_std': df[df['attack']=='1'].groupby(['architecture']).std().accuracy * 100,
        '2_std': df[df['attack']=='2'].groupby(['architecture']).std().accuracy * 100,
        '3_std': df[df['attack']=='3'].groupby(['architecture']).std().accuracy * 100,
    })
    print(final_df)
    # # std_df = pd.DataFrame({
    # #     'full': df.groupby(['architecture']).std().accuracy * 100,
    # #     'fcn': df_fcn.groupby(['architecture']).std().accuracy * 100,
    # #     'conv': df_conv.groupby(['architecture']).std().accuracy * 100
    # # })
    # print(final_df)
    # # print(std_df)
    # # groups = df[df['attack'] == '1'].groupby(['num_param']).mean()
    # # groups = groups.sort_values('num_param')
    # # print(plt.plot(groups.loc['1'].accuracy))
    # # print(groups)
    # # print(groups.accuracy)
    ax = final_df[['1', '2', '3']].plot(kind='bar', color=['#1a9988', '#eb5600', '#6aa4c8'], yerr=final_df[['full_std', 'fcn_std', 'conv_std']].values.T, capsize=4, zorder=3)
    # ax.grid(axis='y', zorder=0)
    # # for p in ax.patches:
    # #     ax.annotate(f'{int(p.get_height())}', (p.get_x() - .01, p.get_height() - .01), fontsize=15)
    # fontsize = 20
    # plt.ylabel('accuracy (%)', fontsize=fontsize)
    # plt.xlabel('shadow model architecture', fontsize=fontsize)
    # ax.tick_params(axis='both', which='major', labelsize=fontsize)
    # plt.title('Convolution vs FCN layers', fontsize=fontsize)
    # plt.ylim(0, 100)
    # plt.legend(fontsize=fontsize)
    # plt.show()


# plot_accuracies_per_architecture_conv()
# plot_accuracies_per_architecture_full()
# plot_accuracies_per_architecture_fcn()
# plot_accuracies_shadows()
plot_accuracies()
plot_accuracies_per_architecture_full()
# plot_accuracies_per_weights()
