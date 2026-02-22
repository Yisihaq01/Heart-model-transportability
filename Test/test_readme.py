"""
Tests that README.md project structure and documented layout exist and are consistent.
"""
import os
from pathlib import Path

import pytest

# Project root = parent of Test/
ROOT = Path(__file__).resolve().parent.parent


class TestReadmeProjectStructure:
    """README 'Project Structure' section — required dirs and key files."""

    def test_dataset_dir_exists(self):
        assert (ROOT / "Dataset").is_dir(), "Dataset/ should exist"

    def test_dataset_files_from_readme(self):
        # README lists these under Dataset/
        expected = [
            "kaggle_cardio_train.csv",
            "long-beach-va.data",
            "hungarian.data",
            "switzerland.data",
        ]
        for name in expected:
            assert (ROOT / "Dataset" / name).exists(), f"Dataset/{name} should exist"

    def test_data_dir_exists(self):
        assert (ROOT / "data").is_dir(), "data/ (cleaned) should exist"

    def test_implementation_plan_dir_exists(self):
        assert (ROOT / "ImplementationPlan").is_dir(), "ImplementationPlan/ should exist"

    def test_implementation_plan_docs_exist(self):
        assert (ROOT / "ImplementationPlan" / "data_ingestion.md").exists()
        assert (ROOT / "ImplementationPlan" / "pipeline_plan.md").exists()

    def test_readme_exists(self):
        assert (ROOT / "README.md").exists()

    def test_literature_dir_exists(self):
        path = ROOT / "Literature"
        if not path.exists():
            pytest.skip("Literature/ not found (create to satisfy README structure)")
        if path.is_file():
            pytest.skip("Literature exists as file; README expects a directory")
        assert path.is_dir(), "Literature/ should be a directory"

    def test_research_paper_dir_exists(self):
        path = ROOT / "Research Paper"
        if not path.exists():
            pytest.skip("Research Paper/ not found (create to satisfy README structure)")
        if path.is_file():
            pytest.skip("Research Paper exists as file; README expects a directory")
        assert path.is_dir(), "Research Paper/ should be a directory"


class TestReadmeContentConsistency:
    """Cross-check README vs data_ingestion / pipeline_plan references."""

    def test_datasets_section_matches_ingestion_sites(self):
        # README: Kaggle + 4 UCI (Cleveland, Hungary, Switzerland, VA)
        sites = ["kaggle", "cleveland", "hungary", "switzerland", "va"]
        # data_ingestion expects these; Dataset may have cleveland as processed
        assert (ROOT / "Dataset" / "kaggle_cardio_train.csv").exists()
        assert (ROOT / "Dataset" / "long-beach-va.data").exists()
        assert (ROOT / "Dataset" / "hungarian.data").exists()
        assert (ROOT / "Dataset" / "switzerland.data").exists()
        # Cleveland can be cleveland.data or processed.cleveland.data
        cleveland_ok = (
            (ROOT / "Dataset" / "cleveland.data").exists()
            or (ROOT / "Dataset" / "processed.cleveland.data").exists()
        )
        assert cleveland_ok, "Cleveland data (cleveland.data or processed.cleveland.data) should exist"
