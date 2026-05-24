"""Tests for French/Italian aliases and pinpoint markers in extract_citation_graph.

Covers:
1. ATF (French) and DTF (Italian) reporters as aliases for BGE.
2. French `c.` and Italian `consid.`/`cons.` pinpoint markers as aliases for `E.`.
3. Modern federal tribunal docket with a space separator (e.g. ``5A 800/2019``).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from extract_citation_graph import (  # noqa: E402  (path setup above)
    classify_docket,
    extract_bge,
    extract_cases,
    extract_references,
)


class BgeReporterAliasTests(unittest.TestCase):
    def test_german_bge_still_extracted(self) -> None:
        refs = extract_bge("Vgl. BGE 137 IV 122 E. 6.2 zur Zustellfiktion.")
        self.assertEqual(len(refs), 1)
        ref = refs[0]
        self.assertEqual(ref.citation, "BGE 137 IV 122 E. 6.2")
        self.assertEqual(ref.segments["reporter"], "BGE")
        self.assertEqual(ref.segments["pinpoint"], "6.2")

    def test_french_atf_normalizes_to_bge(self) -> None:
        refs = extract_bge("Cf. ATF 137 IV 122 c. 6.2 sur la fiction.")
        self.assertEqual(len(refs), 1)
        ref = refs[0]
        # ATF normalizes to BGE, c. normalizes to E.
        self.assertEqual(ref.citation, "BGE 137 IV 122 E. 6.2")
        self.assertEqual(ref.segments["reporter"], "BGE")
        self.assertEqual(ref.segments["pinpoint"], "6.2")
        self.assertEqual(ref.segments["pinpoint_unit"], "E")

    def test_italian_dtf_normalizes_to_bge(self) -> None:
        refs = extract_bge("Vedi DTF 137 IV 122 consid. 6.2 sulla finzione.")
        self.assertEqual(len(refs), 1)
        ref = refs[0]
        self.assertEqual(ref.citation, "BGE 137 IV 122 E. 6.2")
        self.assertEqual(ref.segments["reporter"], "BGE")
        self.assertEqual(ref.segments["pinpoint"], "6.2")

    def test_italian_cons_short_form(self) -> None:
        refs = extract_bge("DTF 139 I 2 cons. 3a")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].citation, "BGE 139 I 2 E. 3a")
        self.assertEqual(refs[0].segments["pinpoint_unit"], "E")

    def test_dedup_across_languages(self) -> None:
        # The same decision in three languages should collapse to one citation
        # string after normalization.
        refs = extract_references(
            "BGE 137 IV 122 E. 6.2 / ATF 137 IV 122 c. 6.2 / DTF 137 IV 122 consid. 6.2"
        )
        bge_refs = [r for r in refs if r.subfamily == "bge"]
        self.assertEqual(len(bge_refs), 1)
        self.assertEqual(bge_refs[0].citation, "BGE 137 IV 122 E. 6.2")


class CaseDocketSeparatorTests(unittest.TestCase):
    def test_underscore_docket_unchanged(self) -> None:
        refs = extract_cases("Urteil 5A_800/2019 vom 12. Februar 2020 E. 3.1")
        self.assertTrue(any(r.citation.startswith("5A_800/2019") for r in refs))

    def test_space_separated_docket_canonicalizes_to_underscore(self) -> None:
        refs = extract_cases("Arret 5A 800/2019 du 12. fevrier 2020 c. 3.1")
        self.assertEqual(len(refs), 1)
        ref = refs[0]
        # Canonical docket uses underscore even when source had a space.
        self.assertTrue(ref.citation.startswith("5A_800/2019"))
        self.assertEqual(ref.segments["docket"], "5A_800/2019")
        self.assertEqual(ref.segments["separator_style"], "space")
        # French c. pinpoint still extracted into the consideration field.
        self.assertEqual(ref.segments["consideration"], "3.1")

    def test_classify_docket_recognises_space_separator(self) -> None:
        subfamily, segments = classify_docket("5A 800/2019")
        self.assertEqual(subfamily, "federal_tribunal_modern_docket")
        self.assertEqual(segments["separator_style"], "space")
        self.assertEqual(segments["court_chamber"], "5")
        self.assertEqual(segments["legal_area_code"], "A")
        self.assertEqual(segments["serial_number"], "800")
        self.assertEqual(segments["decision_year"], "2019")


class ConsiderationMarkerTests(unittest.TestCase):
    def test_french_c_marker_in_modern_docket(self) -> None:
        refs = extract_cases("Arret 1B_210/2023 c. 4.1 sur le recours.")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].segments["consideration"], "4.1")
        self.assertEqual(refs[0].citation, "1B_210/2023 E. 4.1")

    def test_italian_consid_marker_in_modern_docket(self) -> None:
        refs = extract_cases("Sentenza 1B_210/2023 consid. 4.1 sul ricorso.")
        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].segments["consideration"], "4.1")
        self.assertEqual(refs[0].citation, "1B_210/2023 E. 4.1")


if __name__ == "__main__":
    unittest.main()
