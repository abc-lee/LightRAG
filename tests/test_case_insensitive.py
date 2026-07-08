"""测试 entity_name/source/target 提取后统一 lower 化。"""
import pytest
from lightrag.operate import _handle_single_entity_extraction, _handle_single_relationship_extraction


def test_entity_name_lowered():
    """LLM 提取 'Apple'（大写），经 sanitize 后应该是 'apple'（小写）。"""
    record = ["entity", "Apple", "organization", "A fruit company"]
    result = _handle_single_entity_extraction(record, "chunk-test", 1234567890)
    assert result is not None
    assert result["entity_name"] == "apple", f"entity_name 应 lower 化，实际: {result['entity_name']}"
    assert result["entity_type"] == "organization"
    # description 不 lower（自然语言保留大小写）
    assert "fruit" in result["description"].lower() or "fruit" in result["description"]


def test_entity_name_mixed_case_lowered():
    """混合大小写 'XX分行高速公路苏通卡项目' 应该 lower 化。"""
    record = ["entity", "XX分行高速公路苏通卡项目", "project", "某个项目"]
    result = _handle_single_entity_extraction(record, "chunk-test", 1234567890)
    assert result is not None
    assert result["entity_name"] == "xx分行高速公路苏通卡项目"


def test_relationship_source_target_lowered():
    """关系的 source/target 也应该 lower 化。"""
    record = ["relation", "Apple", "Steve Jobs", "founder", "Apple was founded by Steve Jobs"]
    result = _handle_single_relationship_extraction(record, "chunk-test", 1234567890)
    assert result is not None
    assert result["src_id"] == "apple", f"src_id 应 lower 化，实际: {result['src_id']}"
    assert result["tgt_id"] == "steve jobs", f"tgt_id 应 lower 化，实际: {result['tgt_id']}"
    # description 不 lower
    assert "Apple" in result.get("description", "") or "apple" in result.get("description", "").lower()


def test_relationship_self_loop_dropped_after_lower():
    """source='Apple' target='apple' 改后都 lower 成 'apple'，自环检测触发 return None。

    这是 LightRAG 既有语义（_normalize_node_id 早就 lower 化节点 id，自环早就在丢弃），
    v2 只是把 vdb 侧也对齐——不是 v2 引入的新 bug。lower 化后更早触发，跟修复前行为一致。
    """
    record = ["relation", "Apple", "apple", "self", "self relation"]
    result = _handle_single_relationship_extraction(record, "chunk-test", 1234567890)
    assert result is None, "自环关系应被丢弃（source==target after lower）"


async def test_ainsert_custom_kg_entity_name_lowered():
    """ainsert_custom_kg 路径的 entity_name 也应该 lower 化（脑区注入路径）。

    用真实 mock_embedding_func（随机向量，不调用真实 LLM/嵌入模型）+ 真实 vdb 写入，
    验证：传入大写 entity_name "Apple"，写入 GraphML 节点 id 是 "apple"（lower），
    写入 vdb 后 entity_name 字段也是 "apple"（本次修复）。
    不 mock vdb.upsert，让 vdb 真实写入，才能用 get_by_ids 验证字段。
    """
    import tempfile
    import shutil
    import numpy as np
    from unittest.mock import AsyncMock
    from lightrag.lightrag import LightRAG
    from lightrag.base import EmbeddingFunc
    from lightrag.utils import compute_mdhash_id

    async def _raw_embedding_func(texts):
        return np.random.rand(len(texts), 10)

    mock_embedding_func = EmbeddingFunc(
        embedding_dim=10,
        max_token_size=512,
        func=_raw_embedding_func,
    )

    workdir = tempfile.mkdtemp(prefix="lightrag_test_lower_")
    try:
        rag = LightRAG(
            working_dir=workdir,
            llm_model_func=AsyncMock(return_value=""),
            embedding_func=mock_embedding_func,
        )
        await rag.initialize_storages()
        try:
            custom_kg = {
                "entities": [
                    {
                        "entity_name": "Apple",
                        "entity_type": "organization",
                        "description": "A fruit company",
                        "source_id": "chunk-test-1",
                    },
                    {
                        "entity_name": "Steve Jobs",
                        "entity_type": "person",
                        "description": "Founder of Apple",
                        "source_id": "chunk-test-1",
                    },
                ],
                "relationships": [
                    {
                        "src_id": "Apple",
                        "tgt_id": "Steve Jobs",
                        "description": "Apple was founded by Steve Jobs",
                        "keywords": "founder",
                        "source_id": "chunk-test-1",
                    },
                ],
            }

            await rag.ainsert_custom_kg(custom_kg)

            # 检查 GraphML 节点：应该有 "apple" 和 "steve jobs"（lower）
            graph = rag.chunk_entity_relation_graph
            assert await graph.has_node("apple"), "GraphML 应有节点 'apple'（lower）"
            assert await graph.has_node("steve jobs"), "GraphML 应有节点 'steve jobs'（lower）"

            # 关键断言：vdb 里的 entity_name 字段应该是 lower
            # 传入 "Apple" → lower 成 "apple" → compute_mdhash_id("apple") 算出 vdb id
            # 修复前：传入 "Apple" 用大写算 md5，查 "apple" 的 md5 查不到（None）
            # 修复后：传入 "Apple" lower 成 "apple" 再算 md5，查 "apple" 的 md5 能查到
            expected_ent_id = compute_mdhash_id("apple", prefix="ent-")
            vdb_result = await rag.entities_vdb.get_by_ids([expected_ent_id])
            assert vdb_result and len(vdb_result) > 0, \
                f"vdb 应返回结果列表（即使查不到也返回 [None]），实际: {vdb_result}"
            ent_record = vdb_result[0]
            assert ent_record is not None, \
                f"vdb 应有 id={expected_ent_id}（lower 化 entity_name 算出的 md5），但查不到——说明写入时用了大写 'Apple' 的 md5"
            assert ent_record.get("entity_name") == "apple", \
                f"vdb entity_name 应为 'apple'，实际: {ent_record.get('entity_name')}"

            # 关系也检查
            expected_rel_id = compute_mdhash_id("apple" + "steve jobs", prefix="rel-")
            rel_vdb_result = await rag.relationships_vdb.get_by_ids([expected_rel_id])
            assert rel_vdb_result and len(rel_vdb_result) > 0, \
                f"vdb 应返回关系结果列表，实际: {rel_vdb_result}"
            rel_record = rel_vdb_result[0]
            assert rel_record is not None, \
                f"vdb 应有 relation id={expected_rel_id}（lower 化 src/tgt 算出的 md5），但查不到"
            assert rel_record.get("src_id") == "apple", \
                f"vdb src_id 应为 'apple'，实际: {rel_record.get('src_id')}"
            assert rel_record.get("tgt_id") == "steve jobs", \
                f"vdb tgt_id 应为 'steve jobs'，实际: {rel_record.get('tgt_id')}"
        finally:
            await rag.finalize_storages()
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
