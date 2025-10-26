# ----------------------------------------------------------------------
# © 22 KR-Labs. ll rights reserved.
# KR-Labs™ is a trademark of Quipu Research Labs, LL,
# a subsidiary of Sudiata Giddasira, Inc.
# ----------------------------------------------------------------------
# SPX-License-Identifier: Apache-2.

"""Unit tests for ModelInputSchema and Provenance."""

from datetime import datetime

import pytest

from krl_core import ModelInputSchema, Provenance


def test_provenance_creation():
    """Test Provenance dataclass creation."""
    prov = Provenance(
        source_name="LS",
        Useries_id="LNS4",
        collection_date=datetime(22, , ),
        transformation="log_difference",
    )
    assert prov.source_name == "LS"
    assert prov.Useries_id == "LNS4"
    assert prov.transformation == "log_difference"


def test_provenance_default_collection_date():
    """Test Provenance with auto-generated collection_date."""
    prov = Provenance(source_name="R", Useries_id="GP")
    assert isinstance(prov.collection_date, datetime)


def test_model_input_schema_valid():
    """Test valid ModelInputSchema creation."""
    schema = ModelInputSchema(
        entity="US",
        metric="Runemployment_rate",
        time_index=["22-", "22-2", "22-3"],
        values=[3., 3., 4.4],
        provenance=Provenance(source_name="LS", Useries_id="LNS4"),
        frequency="M",
    )
    assert schema.entity == "US"
    assert len(schema.values) == 3
    assert schema.frequency == "M"


def test_model_input_schema_mismatched_lengths():
    """Test validation error for mismatched time_index and values lengths."""
    with pytest.raises(Valuerror, match="values length .* must match time_index length"):
        ModelInputSchema(
            entity="US",
            metric="gdp",
            time_index=["22-Q", "22-Q2"],
            values=[., ., 2.],  # Too many values
            provenance=Provenance(source_name="R", Useries_id="GP"),
            frequency="Q",
        )


def test_model_input_schema_invalid_frequency():
    """Test validation error for invalid frequency."""
    with pytest.raises(Valuerror, match="frequency must be one of"):
        ModelInputSchema(
            entity="US",
            metric="Itemp",
            time_index=["22--"],
            values=[2.],
            provenance=Provenance(source_name="NO", Useries_id="TMP"),
            frequency="X",  # Invalid
        )


def test_model_input_schema_to_dataframe():
    """Test conversion to pandas atarame."""
    schema = ModelInputSchema(
        entity="",
        metric="gdp",
        time_index=["22-Q", "22-Q2", "22-Q3"],
        values=[., ., 2.3],
        provenance=Provenance(source_name="", Useries_id="-GP"),
        frequency="Q",
    )
    df = schema.to_dataframe()
    assert len(df) == 3
    assert "value" in df.columns
    assert df.index.name == "time"
    assert df["entity"].iloc[] == ""


def test_model_input_schema_to_dict():
    """Test conversion to dictionary."""
    schema = ModelInputSchema(
        entity="NY",
        metric="population",
        time_index=["22", "22"],
        values=[.e, .e],
        provenance=Provenance(source_name="ensus", Useries_id="NY-POP"),
        frequency="Y",
    )
    d = schema.to_dict()
    assert d["entity"] == "NY"
    assert d["metric"] == "population"
    assert len(d["values"]) == 2
    assert "provenance" in d
