from tool_index.schema import ToolDescriptor
from tool_index.pipeline.stage2_enrich import enrich_all
from tool_index.providers import FakeLLMProvider


def test_enrich_populates_all_tools():
    llm = FakeLLMProvider()
    descs = [
        ToolDescriptor(id="t1", name="db_users_read", signature="", original_doc="read a user row"),
        ToolDescriptor(id="t2", name="http_post", signature="", original_doc="send a post request"),
    ]
    enrichments = enrich_all(descs, llm, cache=None, batch_size=5)
    assert set(enrichments) == {"t1", "t2"}
    assert enrichments["t1"].intent_phrase
    assert enrichments["t1"].example_queries
