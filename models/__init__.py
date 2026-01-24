"""
models/__init__.py
"""
import importlib
from models.augan_model import AuganModel

def create_model(opt):
    """Create a model given the option."""
    instance = AuganModel(opt)
    print("model [AuganModel] was created")
    return instance

def get_option_setter(model_name):
    """Return the static method <modify_commandline_options> of the model class."""
    # 预防性检查：如果 AuganModel 没有定义这个方法，就返回一个空函数，防止崩溃
    if hasattr(AuganModel, 'modify_commandline_options'):
        return AuganModel.modify_commandline_options
    else:
        return lambda parser, is_train: parser