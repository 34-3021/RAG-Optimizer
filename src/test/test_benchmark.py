import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark import plot_distribution
import numpy as np

# 生成测试数据
scores = [
    {'Chinese': np.random.normal(85, 5), 'English': np.random.normal(80, 8)} 
    for _ in range(50)
]

# 绘制分布图
plot_distribution(scores, "distribution.png")
