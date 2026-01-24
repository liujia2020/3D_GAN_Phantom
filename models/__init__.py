"""
models/__init__.py
"""
import importlib
from models.augan_model import AuganModel

def create_model(opt):
    """创建模型实例"""
    # 强制使用 AuganModel，忽略 opt.model 参数（或者你可以写逻辑判断）
    instance = AuganModel(opt)
    print("model [AuganModel] was created")
    return instance

def get_option_setter(model_name):
    """获取模型特定的参数设置"""
    return AuganModel.modify_commandline_options