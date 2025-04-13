

import argparse

def cal(x, y, cal_mode):
    if cal_mode == 'plus':
        return x + y
    elif cal_mode == 'minus':
        return x - y
    elif cal_mode == 'multiply':
        return x * y
    elif cal_mode == 'divide':
        return x / y
    else:
        raise ValueError("Invalid calculation mode")

if __name__ == "__main__":
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description="Perform calculations based on the mode.")
    
    # 添加命令行参数
    parser.add_argument("--plus", action="store_true", help="Perform addition")
    parser.add_argument("--minus", action="store_true", help="Perform subtraction")
    parser.add_argument("--multiply", action="store_true", help="Perform multiplication")
    parser.add_argument("--divide", action="store_true", help="Perform division")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 设置默认值
    x = 2.0
    y = 3.0
    
    # 根据命令行参数选择计算模式
    if args.plus:
        cal_mode = 'plus'
    elif args.minus:
        cal_mode = 'minus'
    elif args.multiply:
        cal_mode = 'multiply'
    elif args.divide:
        cal_mode = 'divide'
    else:
        raise ValueError("No valid calculation mode specified")
    
    # 执行计算并输出结果
    result = cal(x, y, cal_mode)
    print(f"Result of {cal_mode} operation: {result}")