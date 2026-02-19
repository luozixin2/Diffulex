#!/usr/bin/env python3
"""
DiffuLex Edge - Module entry point

Usage:
    python -m diffulex_edge
    python -m diffulex_edge --help
    python -m diffulex_edge --pte-path model.pte
"""

from .cli import main

if __name__ == "__main__":
    main()
