# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Tuple


def parse_page_options(save_ids: str) -> List[Tuple[int, int]]:
    """
    Parse the page options string and return start and end indices.

    Args:
        save_ids (str): The input string containing page options, e.g., "0,3,4,5,10-20"

    Returns:
        list: A list of tuples containing start and end indices for each range
    """
    # Split the input string into individual ranges
    ranges = save_ids.split(",")
    # Initialize an empty list to store the parsed ranges
    parsed_ranges = []
    # Iterate over each range
    for r in ranges:
        # Check if the range contains a hyphen (i.e., it's a range)
        if "-" in r:
            # Split the range into start and end indices
            start, end = map(int, r.split("-"))
            # Append the parsed range to the list
            parsed_ranges.append((start, end))
        else:
            # If it's not a range, append a tuple with the single index as both start and end
            parsed_ranges.append((int(r), int(r)))
    return parsed_ranges


def is_index_in_range(index: int, ranges: List[Tuple[int, int]]) -> bool:
    """
    Check if an index belongs to any of the given ranges.

    Args:
        index (int): The page index to check
        ranges (list): A list of tuples representing the parsed ranges

    Returns:
        bool: True if the index belongs to any range, False otherwise
    """
    # Iterate over each range
    for start, end in ranges:
        # Check if the index falls within the current range
        if start <= index <= end:
            # If it does, return True immediately
            return True

    # If the index doesn't belong to any range, return False
    return False
