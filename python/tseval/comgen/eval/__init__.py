def get_setup_cls(name: str) -> type:
    if name == "StandardSetup":
        from tseval.comgen.eval.StandardSetup import StandardSetup
        return StandardSetup
    else:
        raise ValueError(f"No setup with name {name}")
