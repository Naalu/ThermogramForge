"""
Debug utilities for ThermogramForge application.
"""

import functools
import json
import logging
import time
from typing import Any, Callable

from dash import callback_context

# Configure logger
logger = logging.getLogger(__name__)


def debug_callback(func: Callable) -> Callable:
    """
    Decorator for debugging callbacks.
    Logs when callback is triggered, what inputs triggered it, and timing information.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        ctx = callback_context
        start_time = time.time()

        # Get trigger info
        triggered = "No triggers" if not ctx.triggered else ctx.triggered[0]["prop_id"]
        logger.info(f"Callback {func.__name__} triggered by: {triggered}")

        # Log input arguments (excluding large data objects)
        arg_info = []
        for i, arg in enumerate(args):
            if isinstance(arg, (str, int, float, bool)):
                arg_info.append(f"arg{i}={arg}")
            elif arg is None:
                arg_info.append(f"arg{i}=None")
            elif hasattr(arg, "shape"):  # For pandas dataframes
                arg_info.append(f"arg{i}=DataFrame[{getattr(arg, 'shape')}]")
            else:
                arg_info.append(f"arg{i}=<{type(arg).__name__}>")

        logger.info(f"Callback {func.__name__} args: {', '.join(arg_info)}")

        try:
            # Run the callback
            result = func(*args, **kwargs)

            # Log completion time and success
            elapsed = time.time() - start_time
            logger.info(f"Callback {func.__name__} completed in {elapsed:.3f}s")

            # Log result info (but not the actual data)
            if isinstance(result, tuple):
                result_info = []
                for i, res in enumerate(result):
                    if res is None:
                        result_info.append(f"result{i}=None")
                    elif isinstance(res, (str, int, float, bool)):
                        if isinstance(res, str) and len(res) > 50:
                            result_info.append(f"result{i}=str({len(res)} chars)")
                        else:
                            result_info.append(f"result{i}={res}")
                    else:
                        result_info.append(f"result{i}=<{type(res).__name__}>")
                logger.info(
                    f"Callback {func.__name__} returned: {', '.join(result_info)}"
                )

            return result

        except Exception as e:
            # Log any errors
            logger.error(f"Callback {func.__name__} error: {str(e)}", exc_info=True)
            raise

    return wrapper


def inspect_store_data(data: Any, name: str = "store") -> None:
    """Inspects and logs the type and basic structure of data from a dcc.Store.

    Attempts to identify if the data is JSON (dict/list) or a simple type,
    logging the keys or length without printing large amounts of data.

    Args:
        data: The data retrieved from a dcc.Store component's 'data' property.
        name: A descriptive name for the store being inspected (for logging).
    """
    if data is None:
        logger.info(f"Store '{name}' contains: None")
        return

    try:
        if isinstance(data, str):
            # Try to parse JSON string
            try:
                parsed = json.loads(data)
                if isinstance(parsed, dict):
                    logger.info(
                        f"Store '{name}' contains JSON dict with keys: {list(parsed.keys())}"
                    )
                elif isinstance(parsed, list):
                    logger.info(
                        f"Store '{name}' contains JSON list with {len(parsed)} items"
                    )
                else:
                    logger.info(f"Store '{name}' contains JSON {type(parsed).__name__}")
            except:
                if len(data) > 100:
                    logger.info(f"Store '{name}' contains string ({len(data)} chars)")
                else:
                    logger.info(f"Store '{name}' contains string: {data}")
        elif isinstance(data, dict):
            logger.info(f"Store '{name}' contains dict with keys: {list(data.keys())}")
        elif isinstance(data, list):
            logger.info(f"Store '{name}' contains list with {len(data)} items")
        else:
            logger.info(f"Store '{name}' contains {type(data).__name__}")
    except Exception as e:
        logger.error(f"Error inspecting store '{name}': {str(e)}")
