import pytest
import os
import tarfile
import json
from pathlib import Path
from scripts.manage_states import sha256_file, DATA_DIR, REQUIRED_TRACKS, get_state_path

def test_sha256_calculation(tmp_path):
    test_file = tmp_path / "sample.bin"
    test_file.write_bytes(b"hello world")
    h = sha256_file(test_file)
    assert h == "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9"

def test_bundle_and_unpack(tmp_path, monkeypatch):
    # Create fake required states
    fake_data = tmp_path / "data"
    fake_data.mkdir()
    monkeypatch.setattr("scripts.manage_states.DATA_DIR", fake_data)

    for tid in REQUIRED_TRACKS:
        p = fake_data / f"Super Mario 64 DS (USA) (Rev 1).{tid}"
        p.write_bytes(f"state content for {tid}".encode())

    bundle_path = tmp_path / "bundle.tar.gz"

    class ArgsBundle:
        out = str(bundle_path)

    from scripts.manage_states import cmd_bundle, cmd_unpack
    ret = cmd_bundle(ArgsBundle())
    assert ret == 0
    assert bundle_path.exists()

    # Clear fake_data files
    for f in fake_data.glob("*.ds*"):
        f.unlink()

    class ArgsUnpack:
        bundle = str(bundle_path)

    ret_unpack = cmd_unpack(ArgsUnpack())
    assert ret_unpack == 0

    for tid in REQUIRED_TRACKS:
        p = fake_data / f"Super Mario 64 DS (USA) (Rev 1).{tid}"
        assert p.exists()
        assert p.read_bytes() == f"state content for {tid}".encode()
