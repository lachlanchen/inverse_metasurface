import spectral

# 定义完整的头文件路径
hdr_file = 'AVIRIS/AV320231008t173943_L2A_OE_main_98b13fff/AV320231008t173943_L2A_OE_main_98b13fff_RFL_ORT.hdr'

# 打开高光谱数据
img = spectral.open_image(hdr_file)
data = img.load()  # 读取数据

# 打印数据形状
print("数据形状:", data.shape)  # 输出: (高度, 宽度, 波段数)

# 获取波长信息
wavelengths = img.bands.centers
print("波长信息:", wavelengths)  # 打印每个波段的波长值

# 计算并打印波长范围
min_wavelength = min(wavelengths)
max_wavelength = max(wavelengths)
print("波长范围: {} - {}".format(min_wavelength, max_wavelength))

