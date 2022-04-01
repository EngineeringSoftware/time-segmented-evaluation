def get_model_cls(name: str) -> type:
    if name == "TransformerACL20":
        from tseval.comgen.model.TransformerACL20 import TransformerACL20
        return TransformerACL20
    elif name == "DeepComHybridESE19":
        from tseval.comgen.model.DeepComHybridESE19 import DeepComHybridESE19
        return DeepComHybridESE19
    else:
        raise ValueError(f"No model with name {name}")
