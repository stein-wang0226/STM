import pandas as pd
import matplotlib.pyplot as plt


def draw(file_paths, data):
    # 计算每个文件的行数
    line_counts = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        line_counts.append(len(df))

    # 绘制曲线图
    plt.plot(range(1, len(file_paths) + 1), line_counts, marker='o')
    plt.xticks(range(1, len(file_paths) + 1), file_paths)
    plt.xlabel('file')
    plt.ylabel('line')
    plt.title(data)

    for i, count in enumerate(line_counts):
        plt.text(i + 1, count, str(count), ha='center', va='bottom')
    plt.show()

# 文件路径
datas = ["Dgraph", "Mooc", "reddit", "wiki"]
ks = ["1", "2", "3", "4", "5", "6"]
for data in datas:
    file_paths = []
    for k in ks:
        file_paths.append(f"{data} result{k}.csv")
    draw(file_paths, data)
