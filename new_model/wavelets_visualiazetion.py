import pywt
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

# 模拟信号
data = np.random.randn(144000)
# 4级小波分解
coeffs = pywt.wavedec(data, wavelet='db4', level=3)
A3, D3, D2, D1 = coeffs

# 打印各系数长度
print("A3长度:", len(A3), "D3长度:", len(D3), "D2长度:", len(D2), "D1长度:", len(D1))

# 小波逆变换重构信号
reconstructed = pywt.waverec(coeffs, wavelet='db4')

# 可视化（分解系数）
plt.figure(figsize=(12, 8))
plt.subplot(5, 1, 1); plt.plot(D1); plt.title("D1 (高频)")
plt.subplot(5, 1, 2); plt.plot(D2); plt.title("D2")
plt.subplot(5, 1, 3); plt.plot(D3); plt.title("D3")
plt.subplot(5, 1, 4); plt.plot(A3); plt.title("A3(最低频)")
plt.subplot(5, 1, 5)
plt.plot(data, label='原始数据', alpha=0.7)
plt.plot(reconstructed, label='逆变换重构数据', alpha=0.7)
plt.title("原始数据 vs 逆变换重构数据")
plt.legend()
plt.tight_layout()
plt.show()