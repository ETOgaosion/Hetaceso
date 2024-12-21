import os
import csv
import colorsys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

fig_path = '../results/fig/parser'
if not os.path.exists(fig_path):
    os.makedirs(fig_path)

def get_hex_color(color_name):
    try:
        return mcolors.CSS4_COLORS[color_name]
    except KeyError:
        raise ValueError(f"Color '{color_name}' is not a valid CSS4 color name")

def generate_similar_colors_with_param(num, base_color, similarity_param=0.1):
    base_color = base_color.lstrip('#')
    r, g, b = tuple(int(base_color[i:i+2], 16) for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r/255.0, g/255.0, b/255.0)
    
    colors = []
    for i in range(num):
        new_h = (h + i * similarity_param / num) % 1.0
        new_r, new_g, new_b = colorsys.hls_to_rgb(new_h, l, s)
        new_color = '#{:02x}{:02x}{:02x}'.format(int(new_r * 255), int(new_g * 255), int(new_b * 255))
        colors.append(new_color)
    
    return colors

blue_colors = generate_similar_colors_with_param(8, get_hex_color('blue'), 0.1)
purple_colors = generate_similar_colors_with_param(8, get_hex_color('purple'), 0.1)

def read_csv_column(file_name, data_types):
    modeling = {}
    realtime = {}
    error = {}
    config = []
    
    for data_type in data_types:
        modeling[data_type] = []
        realtime[data_type] = []
        error[data_type] = []
    
    with open(file_name, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for i, row in enumerate(csv_reader):
            if i % 2 == 0:
                for data_type in data_types:
                    modeling[data_type].append(float(row[data_type]))
            else:
                for data_type in data_types:
                    realtime[data_type].append(float(row[data_type]))
                    error[data_type].append(abs(realtime[data_type][-1] - modeling[data_type][-1]) / realtime[data_type][-1])
                config.append(row['config'])
    
    return {'modeling': modeling, 'realtime': realtime, 'error': error, 'config': config}

def collect_data_from_csv(data_types):
    data = {}
    directory = '../results/parser'
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory, file_name)
            column_data = read_csv_column(file_path, data_types)
            data[os.path.splitext(file_name)[0]] = column_data
    
    return data

def plot_data_datatype(key, data, data_type):
    print(f'ploting {key} {data_type}')
    modeling = data['modeling'][data_type]
    realtime = data['realtime'][data_type]
    config = data['config']
    
    x = range(len(modeling))
    
    fig, ax = plt.subplots()
    
    bars1 = ax.bar([i * 2 for i in x], modeling, color=blue_colors, width=0.5)
    bars2 = ax.bar([i * 2 + 0.5 for i in x], realtime, color=purple_colors, width=0.5)
    if data_type == "total_time":
        for bar1, bar2, cfg in zip(bars1, bars2, config):
            bar1.set_label("and")
            bar2.set_label(cfg)
    
    ax.set_xlabel('Configurations')
    ax.set_ylabel('Total Time')
    ax.set_xticks([i * 2 + 0.25 for i in x], [f'{c}' for c in range(len(config))])
    ax.set_title(f'Total Time for {key}')
    ax.set_ylim(0, max(max(modeling), max(realtime)) * 1.1)
    
    if data_type == "total_time":
        ax.legend(loc='upper left', ncol=2, columnspacing=1, fontsize='small')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'{key}-{data_type}.png'), dpi=1000)

def plot_data(key, data, data_types):
    for data_type in data_types:
        plot_data_datatype(key, data, data_type)

def plot_error(key, data, data_types):
    print(f'ploting {key} error')
    errors = []
    for data_type in data_types:
        errors.append(data['error'][data_type])
    config = data['config']
    red_colors = generate_similar_colors_with_param(len(data_types), get_hex_color('red'), 1)
    
    x = range(len(errors[0]))
    
    fig, ax = plt.subplots()
    
    for i, error in enumerate(errors):
        line = ax.plot(x, error, color=red_colors[i], label=data_types[i])
        
    max_errors = [max(e) for e in errors]
    max_error = max(max_errors)
    line = ax.axhline(y=max_error, color='black', linestyle='--', label='max error')
    error_larger_than_10_percent = [data_types[i] for i, e in enumerate(max_errors) if e > 0.1]
    
    ax.set_xlabel('Configurations')
    ax.set_ylabel('Error')
    ax.set_xticks(x, [f'{c}' for c in range(len(config))])
    ax.set_title(f'Error for {key}')
    ax.set_ylim(0, 1)
    
    lgd = ax.legend(loc="upper center", ncol=3, columnspacing=1, fontsize='small')
    for text, label in zip(lgd.get_texts(), data_types):
        if label in error_larger_than_10_percent:
            text.set_color('red')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_path, f'{key}-error-total.png'), dpi=1000)

data = collect_data_from_csv(["total_time", "fwd_time", "bwd_time", "embed_fwd_time", "attn_fwd_time", "mlp_fwd_time", "post_fwd_time", "memory_sum", "ref_total_memory"])
for key, value in data.items():
    # plot_data(key, value, ["total_time", "memory_sum", "ref_total_memory"])
    plot_error(key, value, ["total_time", "fwd_time", "bwd_time", "embed_fwd_time", "attn_fwd_time", "mlp_fwd_time", "post_fwd_time", "memory_sum", "ref_total_memory"])
