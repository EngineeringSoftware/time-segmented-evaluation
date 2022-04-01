def get_model_cls(name: str) -> type:
    if name == "Code2SeqICLR19":
        from tseval.metnam.model.Code2SeqICLR19 import Code2SeqICLR19
        return Code2SeqICLR19
    elif name == "Code2VecPOPL19":
        from tseval.metnam.model.Code2VecPOPL19 import Code2VecPOPL19
        return Code2VecPOPL19
    else:
        raise ValueError(f"No model with name {name}")
