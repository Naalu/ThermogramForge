from typing import Any

# This is a minimal stub file for rpy2.robjects to satisfy type checking

class FloatVector:
    def __init__(self, obj: Any) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...

class IntVector:
    def __init__(self, obj: Any) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...

class StrVector:
    def __init__(self, obj: Any) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...

class BoolVector:
    def __init__(self, obj: Any) -> None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, key: Any) -> Any: ...

class r:
    @staticmethod
    def __call__(string: str) -> Any: ...

class packages:
    @staticmethod
    def importr(name: str) -> Any: ...

class numpy2ri:
    @staticmethod
    def activate() -> None: ...
    @staticmethod
    def deactivate() -> None: ...
