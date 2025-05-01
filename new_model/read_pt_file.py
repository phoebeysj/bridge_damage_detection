import torch
import os
import sys

def read_pt_file(file_path):
    """读取.pt文件并打印其内容结构"""
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    try:
        # 加载.pt文件
        data = torch.load(file_path)
        
        # 打印数据类型
        print(f"\n文件: {os.path.basename(file_path)}")
        print(f"数据类型: {type(data)}")
        
        # 如果是字典，打印键
        if isinstance(data, dict):
            print(f"包含的键: {list(data.keys())}")
            
            # 对每个键，打印对应值的类型和形状（如果是张量）
            for key, value in data.items():
                if isinstance(value, torch.Tensor):
                    print(f"  - {key}: {type(value)} 形状: {value.shape}")
                else:
                    print(f"  - {key}: {type(value)}")
        
        # 如果是张量，打印形状
        elif isinstance(data, torch.Tensor):
            print(f"张量形状: {data.shape}")
        
        # 其他情况
        else:
            print("数据不是字典或张量")
        
        return data
    
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("请提供.pt文件路径")
        print("用法: python read_pt_file.py <文件路径>")
        exit(1)
    
    file_path = sys.argv[1]
    read_pt_file(file_path) 