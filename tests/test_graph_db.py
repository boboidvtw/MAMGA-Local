"""
Unit tests for memory/graph_db.py

Tests cover:
- Node insertion and retrieval
- Link creation and traversal
- Serialisation round-trips (to_dict / from_dict)
- Edge-cases: duplicate nodes, missing nodes, empty graph
"""
import pytest
from datetime import datetime

from memory.graph_db import (
    NetworkXGraphDB,
    EventNode,
    EpisodeNode,
    Link,
    NodeType,
    LinkType,
    LinkSubType,
)


# ---------------------------------------------------------------------------
# Node operations
# ---------------------------------------------------------------------------

class TestNodeOperations:
    def test_add_and_get_event_node(self, graph_db, event_node):
        graph_db.add_node(event_node)
        retrieved = graph_db.get_node(event_node.node_id)
        assert retrieved is not None
        assert retrieved.node_id == event_node.node_id
        assert retrieved.content_narrative == event_node.content_narrative

    def test_get_nonexistent_node_returns_none(self, graph_db):
        assert graph_db.get_node("does-not-exist") is None

    def test_add_duplicate_node_id_overwrites(self, graph_db, event_node):
        graph_db.add_node(event_node)
        updated = EventNode(
            node_id=event_node.node_id,
            content_narrative="Updated narrative",
        )
        graph_db.add_node(updated)
        retrieved = graph_db.get_node(event_node.node_id)
        assert retrieved.content_narrative == "Updated narrative"

    def test_add_episode_node(self, graph_db):
        ep = EpisodeNode(node_id="ep-001", title="Trip to Paris", summary="Alice's Paris trip")
        graph_db.add_node(ep)
        retrieved = graph_db.get_node("ep-001")
        assert retrieved is not None
        assert retrieved.title == "Trip to Paris"

    def test_node_count_increases(self, graph_db, event_node, another_event_node):
        initial = graph_db.graph.number_of_nodes()
        graph_db.add_node(event_node)
        graph_db.add_node(another_event_node)
        assert graph_db.graph.number_of_nodes() == initial + 2


# ---------------------------------------------------------------------------
# Link operations
# ---------------------------------------------------------------------------

class TestLinkOperations:
    def test_add_link_and_get_neighbors(self, graph_db, event_node, another_event_node, temporal_link):
        graph_db.add_node(event_node)
        graph_db.add_node(another_event_node)
        graph_db.add_link(temporal_link)

        neighbors = graph_db.get_neighbors(another_event_node.node_id)
        neighbor_ids = [n.node_id for n, _ in neighbors]
        assert event_node.node_id in neighbor_ids

    def test_get_neighbors_with_link_type_filter(self, graph_db, event_node, another_event_node, temporal_link):
        graph_db.add_node(event_node)
        graph_db.add_node(another_event_node)
        graph_db.add_link(temporal_link)

        temporal_neighbors = graph_db.get_neighbors(
            another_event_node.node_id, link_type=LinkType.TEMPORAL
        )
        semantic_neighbors = graph_db.get_neighbors(
            another_event_node.node_id, link_type=LinkType.SEMANTIC
        )

        assert len(temporal_neighbors) == 1
        assert len(semantic_neighbors) == 0

    def test_no_neighbors_on_isolated_node(self, graph_db, event_node):
        graph_db.add_node(event_node)
        assert graph_db.get_neighbors(event_node.node_id) == []

    def test_link_properties_preserved(self, graph_db, event_node, another_event_node):
        graph_db.add_node(event_node)
        graph_db.add_node(another_event_node)
        link = Link(
            source_node_id=event_node.node_id,
            target_node_id=another_event_node.node_id,
            link_type=LinkType.SEMANTIC,
            properties={"similarity": 0.92},
        )
        graph_db.add_link(link)
        _, returned_link = graph_db.get_neighbors(event_node.node_id)[0]
        assert returned_link.properties.get("similarity") == pytest.approx(0.92)


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

class TestSerialisation:
    def test_event_node_round_trip(self, event_node):
        d = event_node.to_dict()
        restored = EventNode.from_dict(d)
        assert restored.node_id == event_node.node_id
        assert restored.content_narrative == event_node.content_narrative
        assert restored.node_type == event_node.node_type
        assert restored.timestamp == event_node.timestamp

    def test_episode_node_round_trip(self):
        ep = EpisodeNode(
            node_id="ep-002",
            title="Summer holiday",
            summary="Family trip.",
            start_timestamp=datetime(2024, 7, 1),
            end_timestamp=datetime(2024, 7, 14),
            event_count=5,
        )
        d = ep.to_dict()
        restored = EpisodeNode.from_dict(d)
        assert restored.node_id == ep.node_id
        assert restored.title == ep.title
        assert restored.event_count == ep.event_count
        assert restored.start_timestamp == ep.start_timestamp

    def test_event_node_missing_timestamp(self):
        """from_dict should not crash when timestamp is absent."""
        data = {"node_id": "x", "content_narrative": "test"}
        node = EventNode.from_dict(data)
        assert node.node_id == "x"
