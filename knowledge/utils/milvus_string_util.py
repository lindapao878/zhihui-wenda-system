"""Milvus 过滤表达式字符串安全转义工具。

Milvus filter 表达式字符串统一转义工具。
"""


def escape_milvus_string(value: str) -> str:
    """
    Milvus过滤表达式专用字符串安全转义函数
    核心作用：
        避免因原始字符串含特殊字符，导致Milvus解析filter_expr时报错，保证CRUD操作正常执行
    转义规则：
        1. 反斜杠（\）→ 双反斜杠（\\）：Milvus表达式转义规则
        2. 双引号（"）→ 转义双引号（\"）：避免截断字符串表达式
        3. 换行/回车/制表符 → 空格：防止表达式换行导致解析失败
    """
    if value is None:
        return ""
    # 确保输入为字符串类型，避免非字符串值报错
    s = str(value)
    # 按Milvus规则转义特殊字符
    s = s.replace("\\", "\\\\").replace('"', '\\"')
    # 替换换行/回车/制表符为空格，保证表达式单行有效
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return s
