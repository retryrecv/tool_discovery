from pathlib import Path

from tool_index.pipeline import build_tree_index


def test_mini_pipeline_builds_valid_tree(config, mini_tools, tmp_path):
    tree = build_tree_index(mini_tools, config, out_root=tmp_path / "snapshots", strict=False)

    # Depth = 4 or 5 (root + L1 [+ L2] + L3 + tool leaf). With the mini fixture
    # there are only a handful of coarse domains so the orchestrator may relabel
    # the category level as domain — which the page explicitly supports.
    assert tree.depth() in (4, 5)

    # All nodes have a parent (except root) and parent exists in tree
    for n in tree.all_nodes():
        if n.id == tree.root.id:
            continue
        assert n.parent_id is not None
        assert n.parent_id in tree.nodes_by_id

    # All tools reachable from root
    reached = set()

    def walk(node):
        for cid in node.children:
            if cid in tree.nodes_by_id:
                walk(tree.nodes_by_id[cid])
            else:
                reached.add(cid)

    walk(tree.root)
    assert reached == set(tree.tools_by_id)

    # Snapshot was written
    vdir = tmp_path / "snapshots" / tree.version
    assert (vdir / "tree.json").exists()
    assert (vdir / "build_trace.json").exists()
    assert (vdir / "seed_eval_set.jsonl").exists()
