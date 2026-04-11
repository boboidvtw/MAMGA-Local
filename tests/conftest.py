"""
Shared pytest fixtures for MAMGA-Local test suite.
"""
import os
import sys
from datetime import datetime

import pytest

# Make project root importable without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Graph DB fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def graph_db():
    """Fresh NetworkXGraphDB instance for each test."""
    from memory.graph_db import NetworkXGraphDB
    return NetworkXGraphDB()


@pytest.fixture
def event_node():
    """A sample EventNode."""
    from memory.graph_db import EventNode, NodeType
    return EventNode(
        node_id="node-001",
        node_type=NodeType.EVENT,
        content_narrative="Alice visited Paris on Monday.",
        timestamp=datetime(2024, 3, 11, 10, 0),
        attributes={"speaker": "Alice"},
    )


@pytest.fixture
def another_event_node():
    """A second EventNode to create links against."""
    from memory.graph_db import EventNode, NodeType
    return EventNode(
        node_id="node-002",
        node_type=NodeType.EVENT,
        content_narrative="Bob booked flights to Paris.",
        timestamp=datetime(2024, 3, 10, 8, 0),
        attributes={"speaker": "Bob"},
    )


@pytest.fixture
def temporal_link(event_node, another_event_node):
    """A TEMPORAL link from node-002 → node-001."""
    from memory.graph_db import Link, LinkType
    return Link(
        link_id="link-001",
        source_node_id=another_event_node.node_id,
        target_node_id=event_node.node_id,
        link_type=LinkType.TEMPORAL,
        properties={"weight": 1.0},
    )


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def base_date():
    """A fixed reference datetime for temporal parsing tests."""
    return datetime(2024, 5, 15, 12, 0, 0)
