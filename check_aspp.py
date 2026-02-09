import torch
# 确保你在项目根目录下运行，否则路径会报错
from models.networks.generator import ResUnetGenerator

print("正在创建生成器模型...")
# 模拟 Exp 50 的参数：开启 use_aspp=True
net = ResUnetGenerator(input_nc=1, output_nc=1, ngf=64, use_aspp=True)

print("\n====== 检查最内层结构 ======")
# 递归打印最内层的结构，看看有没有 ASPP
def find_innermost(module):
    for name, child in module.named_children():
        # 如果找到 ASPP3D 类，直接报告成功
        if "ASPP3D" in str(type(child)):
            print(f"✅ 成功找到 ASPP 模块: {child}")
            return True
        # 否则继续递归
        if find_innermost(child):
            return True
    return False

found = find_innermost(net)

if found:
    print("\n🎉 验证通过！空洞卷积 (ASPP) 已生效。")
else:
    print("\n❌ 验证失败！未找到 ASPP 模块，请检查 use_aspp 参数是否传递正确。")