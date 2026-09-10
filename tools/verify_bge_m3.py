"""验证 BGE-M3 encode_documents 返回的 dense/sparse 结构是否正常。

用法:
    .venv\Scripts\python.exe tools\verify_bge_m3.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge.utils.bge_m3_embedding_util import get_beg_m3_embedding_model


def main():
    print("加载 BGE-M3 模型...")
    model = get_beg_m3_embedding_model()
    if model is None:
        print("FAIL: 模型加载失败")
        sys.exit(1)

    test_texts = ["测试文本", "余华《活着》"]
    print(f"编码 {len(test_texts)} 条文本...")
    result = model.encode_documents(test_texts)

    dense = result.get("dense")
    sparse = result.get("sparse")

    if dense is None:
        print("FAIL: dense 为 None")
        sys.exit(1)

    print(f"dense shape: {dense.shape}")
    assert dense.shape[0] == len(test_texts), f"dense 行数应为 {len(test_texts)}，实际 {dense.shape[0]}"
    assert dense.shape[1] == 1024, f"dense 维度应为 1024，实际 {dense.shape[1]}"

    if sparse is None:
        print("FAIL: sparse 为 None")
        sys.exit(1)

    print(f"sparse shape: {sparse.shape}")
    print(f"sparse indptr: {sparse.indptr}")
    print(f"sparse data len: {len(sparse.data)}")
    print(f"sparse indices len: {len(sparse.indices)}")

    assert sparse.shape[0] == len(test_texts), (
        f"sparse 行数应为 {len(test_texts)}，实际 {sparse.shape[0]}"
    )
    assert len(sparse.indptr) == len(test_texts) + 1, (
        f"indptr 长度应为 {len(test_texts) + 1}，实际 {len(sparse.indptr)}"
    )

    if len(sparse.data) == 0:
        print("WARN: 稀疏向量为空（可能 tokenizer 不适配），但结构正确，不会越界")
    else:
        print("OK: 稀疏向量有数据")

    print("\n全部检查通过 ✅")


if __name__ == "__main__":
    main()
