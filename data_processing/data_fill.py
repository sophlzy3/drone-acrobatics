import pandas as pd
import numpy as np

def fill_trajectory_fields(input_csv, output_csv):
    # Use low_memory=False to handle the mixed types warning
    df = pd.read_csv(input_csv, low_memory=False)
    time_col = 'timestamp'
    
    # 清理时间戳数据
    try:
        # 首先尝试直接转换为数值
        df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
        
        # 检查是否有无效值
        invalid_rows = df[df[time_col].isna()]
        if not invalid_rows.empty:
            print(f"警告：发现{len(invalid_rows)}行无效的时间戳数据")
            # 删除包含无效时间戳的行
            df = df.dropna(subset=[time_col])
            print(f"已删除无效行，剩余{len(df)}行数据")
    except Exception as e:
        print(f"处理时间戳时出错：{str(e)}")
        raise
    
    timestamps = df[time_col].values
    
    # 检查数据点数量
    if len(timestamps) <= 2:
        raise ValueError("需要至少3个数据点才能进行微分计算")

    def finite_diff(data, t):
        result = np.zeros_like(data)
        # 对内部点使用中心差分
        for i in range(1, len(t)-1):
            dt = (t[i+1] - t[i-1])
            if dt > 0:  # 防止除以零
                result[i] = (data[i+1] - data[i-1]) / dt
        
        # 对第一个点使用前向差分
        if len(t) > 1 and (t[1] - t[0]) > 0:
            result[0] = (data[1] - data[0]) / (t[1] - t[0])
            
        # 对最后一个点使用后向差分
        if len(t) > 1 and (t[-1] - t[-2]) > 0:
            result[-1] = (data[-1] - data[-2]) / (t[-1] - t[-2])
            
        return result

    def moving_avg(data, window=10):
        return pd.DataFrame(data).rolling(window=window, min_periods=1, center=True).mean().values

    # === Fields ===
    linear_vel_cols = ['velocity.linear.x', 'velocity.linear.y', 'velocity.linear.z']
    angular_vel_cols = ['velocity.angular.x', 'velocity.angular.y', 'velocity.angular.z']

    linear_acc_cols = ['acceleration.linear.x', 'acceleration.linear.y', 'acceleration.linear.z']
    angular_acc_cols = ['acceleration.angular.x', 'acceleration.angular.y', 'acceleration.angular.z']

    acc_bias_cols = ['acc_bias.x', 'acc_bias.y', 'acc_bias.z']
    gyr_bias_cols = ['gyr_bias.x', 'gyr_bias.y', 'gyr_bias.z']

    jerk_cols = ['jerk.x', 'jerk.y', 'jerk.z']
    snap_cols = ['snap.x', 'snap.y', 'snap.z']

    try:
        # === Step 1: Acceleration ===
        df[linear_acc_cols] = finite_diff(df[linear_vel_cols].values, timestamps)
        df[angular_acc_cols] = finite_diff(df[angular_vel_cols].values, timestamps)

        # === Step 2: Jerk ===
        jerk_data = finite_diff(df[linear_acc_cols].values, timestamps)
        df[jerk_cols] = jerk_data

        # === Step 3: Snap ===
        snap_data = finite_diff(jerk_data, timestamps)
        df[snap_cols] = snap_data

        # === Step 4: Biases (smoothed from existing signals) ===
        df[acc_bias_cols] = moving_avg(df[linear_acc_cols].values)
        df[gyr_bias_cols] = moving_avg(df[angular_acc_cols].values)

        # === Save ===
        df.to_csv(output_csv, index=False)
        print(f"Finish processing, saved to {output_csv}")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise