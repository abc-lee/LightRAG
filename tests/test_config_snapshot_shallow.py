"""验证 LightRAG._config_snapshot 是浅拷贝（不对字段值做深拷贝）。

背景：原实现使用 ``dataclasses.asdict(self)``,其标准库语义是递归深拷贝整个对象图，
每次查询/每次入库都会把 tokenizer 等重对象复制一遍（实测单次 1.85s）。
新实现只新建顶层 dict，字段值保持同一引用。
"""

from __future__ import annotations

from dataclasses import dataclass

from lightrag.lightrag import LightRAG


class _DeepcopyCountingToken:
    """充当 HF tokenizer 等重对象：统计 __deepcopy__ 被调用的次数。"""

    def __init__(self) -> None:
        self.deepcopy_calls = 0

    def __deepcopy__(self, memo):
        self.deepcopy_calls += 1
        return self


@dataclass
class _DummyConfig:
    """模拟 LightRAG 字段形状的 dummy dataclass。"""

    workspace: str = "test"
    tokenizer: object = None
    child: object = None


def test_config_snapshot_returns_dict_of_same_references():
    dummy = _DummyConfig(
        workspace="ws",
        tokenizer=_DeepcopyCountingToken(),
        child=_DummyConfig(workspace="inner"),
    )
    snapshot = LightRAG._config_snapshot(dummy)
    assert isinstance(snapshot, dict)
    assert snapshot["workspace"] == "ws"
    assert snapshot["tokenizer"] is dummy.tokenizer
    assert snapshot["child"] is dummy.child


def test_config_snapshot_does_not_deepcopy_field_values():
    token = _DeepcopyCountingToken()
    dummy = _DummyConfig(tokenizer=token)
    LightRAG._config_snapshot(dummy)
    assert token.deepcopy_calls == 0


def test_config_snapshot_creates_fresh_dict_each_call():
    dummy = _DummyConfig(workspace="ws")
    s1 = LightRAG._config_snapshot(dummy)
    s2 = LightRAG._config_snapshot(dummy)
    assert s1 is not s2
    assert s1.keys() == s2.keys()
