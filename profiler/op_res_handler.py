import os
import csv
import matplotlib.pyplot as plt
from model_configs import model_prof_configs

data = {
    # model_size
        # op
            # tp
                # mbs
                    # seqlen
                        # fwd_time
                        # bwd_time
}

model_size = ['350M']
tp = [1]


def get_data(file, model_size, tp, mbs, seqlen):
    with open(f'../results/profiled-gpt-hetaceso/{file}', newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0] == 'op_name':
                continue
            if model_size not in data:
                data[model_size] = {}
            if row[0] not in data[model_size]:
                data[model_size][row[0]] = {}
            if tp not in data[model_size][row[0]]:
                data[model_size][row[0]][tp] = {}
            if mbs not in data[model_size][row[0]][tp]:
                data[model_size][row[0]][tp][mbs] = {}
            if seqlen not in data[model_size][row[0]][tp][mbs]:
                data[model_size][row[0]][tp][mbs][seqlen] = {}
            data[model_size][row[0]][tp][mbs][seqlen] = {'fwd_time': float(row[1]), 'bwd_time': float(row[2])}

def get_all_data():
    for model in model_size:
        for tp_size in tp:
            for mbs in model_prof_configs['gpt']['mbs'][model]:
                for seqlen in model_prof_configs['gpt']['seqlen']:
                    get_data(f'gpt_{model}_mbs{mbs}_seqlen{seqlen}_tp{tp_size}.csv', model, tp_size, mbs, seqlen)

get_all_data()
print('============ finish data handling =============')

def plot_single(data, file_path):
    fwd_time = []
    bwd_time = []
    seqlen = []
    for s in data:
        seqlen.append(s)
        fwd_time.append(data[s]['fwd_time'])
        bwd_time.append(data[s]['bwd_time'])
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=1000)
    axes[0].plot(seqlen, fwd_time)
    axes[0].set_title('Forward Time')
    axes[0].set_xlabel('Sequence Length')
    axes[0].set_ylabel('Time (us)')
    max_0 = int(max(fwd_time))
    axes[0].set_yticks(range(int(min(fwd_time)), max_0, max_0 // 10))
    axes[1].plot(seqlen, bwd_time)
    axes[1].set_title('Backward Time')
    axes[1].set_xlabel('Sequence Length')
    axes[1].set_ylabel('Time (us)')
    max_1 = int(max(bwd_time))
    axes[1].set_yticks(range(int(min(bwd_time)), max_1, max_1 // 10))
    plt.savefig(f'{file_path}.png')

def plot_data(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    for model in data:
        for op in data[model]:
            for tp_size in data[model][op]:
                for mbs in data[model][op][tp_size]:
                    plot_single(data[model][op][tp_size][mbs], f'{dir}/{model}_{op}_tp{tp_size}_mbs{mbs}')
                    print(f'============ finish plotting {model}_{op}_tp{tp_size}_mbs{mbs} =============')

plot_data('../results/fig/profiled-gpt-hetaceso')