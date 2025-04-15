"""
Utility for checking layout for duplicate IDs.
"""

import logging
from collections import defaultdict
from typing import Any, DefaultDict, Dict

logger = logging.getLogger(__name__)


def find_duplicate_ids(layout: Any) -> Dict[str, int]:
    """
    Find duplicate IDs in a Dash layout.

    Args:
        layout: The Dash layout object to check

    Returns:
        A dict mapping duplicate IDs to their count
    """
    if not layout:
        logger.warning("Empty layout provided to duplicate ID checker")
        return {}

    ids: DefaultDict[str, int] = defaultdict(int)
    _extract_ids(layout, ids)

    # Filter for IDs that appear more than once
    duplicates: Dict[str, int] = {id: count for id, count in ids.items() if count > 1}

    if duplicates:
        logger.error(f"Found duplicate IDs in layout: {duplicates}")
    else:
        logger.info("No duplicate IDs found in layout")

    return duplicates


def _extract_ids(component: Any, id_counter: DefaultDict[str, int]) -> None:
    """
    Recursively extract component IDs from a layout.

    Args:
        component: A Dash component
        id_counter: Dict to count occurrences of each ID
    """
    # Check if component has an ID
    component_id = getattr(component, "id", None)
    if component_id is not None:
        id_counter[component_id] += 1

    # Check for children
    children = getattr(component, "children", None)
    if children:
        if isinstance(children, list):
            for child in children:
                _extract_ids(child, id_counter)
        else:
            _extract_ids(children, id_counter)
