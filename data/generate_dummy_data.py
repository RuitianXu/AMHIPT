import pandas as pd
import numpy as np

# ================= 配置区域 =================
# 生成的样本总数
NUM_SAMPLES = 100 

# 定义数据中的主要物种及其大概的形态范围 (最小值, 最大值)
SPECIES_CONFIG = {
    'Ficedula zanthopygia': { # 白眉姬鹟
        'Beak': (12.0, 15.0), 'HeadBill': (30.0, 33.0), 
        'Wing': (66.0, 74.0), 'Tail': (43.0, 50.0), 'Weight': (11.0, 14.0)
    },
    'Parus minor': { # 远东山雀
        'Beak': (8.0, 12.0), 'HeadBill': (26.0, 32.0), 
        'Wing': (60.0, 72.0), 'Tail': (50.0, 70.0), 'Weight': (10.0, 16.0)
    },
    'Poecile montanus': { # 褐头山雀
        'Beak': (7.0, 11.0), 'HeadBill': (25.0, 29.0), 
        'Wing': (57.0, 67.0), 'Tail': (50.0, 65.0), 'Weight': (9.0, 13.0)
    },
    'Tarsiger cyanurus': { # 红胁蓝尾鸲
        'Beak': (7.0, 12.0), 'HeadBill': (29.0, 33.0), 
        'Wing': (72.0, 84.0), 'Tail': (50.0, 60.0), 'Weight': (11.0, 15.0)
    }
}

# ===========================================

def generate_example_data():
    data_rows = []
    species_keys = list(SPECIES_CONFIG.keys())

    for _ in range(NUM_SAMPLES):
        # 1. 随机选择一个物种
        sp_name = np.random.choice(species_keys)
        ranges = SPECIES_CONFIG[sp_name]

        # 2. 根据物种的范围生成随机形态数据
        # 加上 round(2) 保留两位小数
        row = {
            '年份': np.random.choice([2016, 2017, 2021]),
            '采集地点': 'Example_Location',
            '种名': sp_name,
            '原始编号': f'EX-{np.random.randint(1000, 9999)}',
            # 特征
            '喙长mm': round(np.random.uniform(ranges['Beak'][0], ranges['Beak'][1]), 2),
            '头喙mm': round(np.random.uniform(ranges['HeadBill'][0], ranges['HeadBill'][1]), 2),
            '翼长mm': round(np.random.uniform(ranges['Wing'][0], ranges['Wing'][1]), 2),
            '尾长mm': round(np.random.uniform(ranges['Tail'][0], ranges['Tail'][1]), 2),
            '体重g':  round(np.random.uniform(ranges['Weight'][0], ranges['Weight'][1]), 2),
            # 虽然模型不用跗跖长，但为了模拟原始数据结构还是要加上
            '跗跖长mm': round(np.random.uniform(15.0, 25.0), 2),
            
            # 随机生成感染状态
            '感染状态': np.random.choice(['感染', '未感染'], p=[0.3, 0.7]), # 假设30%感染率
            
            # 其他数据
            'lineage名': np.nan,
            '寄生虫属': np.nan
        }
        data_rows.append(row)

    # 创建 DataFrame
    df = pd.DataFrame(data_rows)
    
    # 保存为 Excel
    output_filename = 'example_data.xlsx'
    df.to_excel(output_filename, index=False)
    
    print(f"✅ 成功生成虚拟示例数据: {output_filename}")
    print(f"   包含了 {NUM_SAMPLES} 条样本，涵盖了4个主要物种。")

if __name__ == "__main__":
    generate_example_data()
