from tool_index.clustering import agglomerative_cluster, rebalance_clusters


def test_agglomerative_groups_similar():
    # Two obvious groups: first dim strong, last dim strong
    embs = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.95, 0.05, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.1, 0.9],
    ]
    clusters = agglomerative_cluster(embs, distance_threshold=0.2)
    assert len(clusters) == 2
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 3]


def test_rebalance_splits_oversized():
    # 8 points all in one cluster; min/max = (2, 4)
    embs = [[float(i), 0.0] for i in range(8)]
    clusters = [list(range(8))]
    out = rebalance_clusters(clusters, embs, min_size=2, max_size=4)
    assert all(len(c) <= 4 for c in out)
    assert sum(len(c) for c in out) == 8
