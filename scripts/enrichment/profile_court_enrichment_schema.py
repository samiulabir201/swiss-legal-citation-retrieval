#!/usr/bin/env python
"""Compatibility wrapper for the renamed court enrichment profiler."""

from __future__ import annotations

from court_enrichment_profile import main


if __name__ == "__main__":
    raise SystemExit(main())
