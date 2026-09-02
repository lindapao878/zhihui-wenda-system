"""Knowledge query exceptions."""


class StateFieldError(Exception):
    def __init__(self, node_name: str, field_name: str, expected_type):
        self.node_name = node_name
        self.field_name = field_name
        self.expected_type = expected_type
        super().__init__(
            f"[{node_name}] 状态字段 '{field_name}' 错误，期望类型: {expected_type}"
        )
