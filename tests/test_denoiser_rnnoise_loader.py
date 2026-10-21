"""Locating librnnoise, and what to say when it is not there.

``find_library`` covers the system linker paths only. A source build and a
Homebrew install both land outside them, and the Homebrew package named
"rnnoise" is a cask of DAW plugins that cannot satisfy this C ABI — so the
guidance the ImportError carries has to send the reader somewhere real.
"""

from __future__ import annotations

import os

import pytest

from roomkit.voice.pipeline.denoiser import rnnoise


class TestSearchDirs:
    def test_probes_the_homebrew_lib_dir(self) -> None:
        assert "/opt/homebrew/lib" in rnnoise._rnnoise_search_dirs()

    def test_honours_an_explicit_homebrew_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("HOMEBREW_PREFIX", "/custom/brew")
        assert "/custom/brew/lib" in rnnoise._rnnoise_search_dirs()

    def test_keeps_the_source_build_prefixes(self) -> None:
        dirs = rnnoise._rnnoise_search_dirs()
        assert os.path.expanduser("~/.local/lib") in dirs
        assert "/usr/local/lib" in dirs

    def test_lists_each_directory_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An exported HOMEBREW_PREFIX usually IS the default prefix."""
        monkeypatch.setenv("HOMEBREW_PREFIX", "/opt/homebrew")
        dirs = rnnoise._rnnoise_search_dirs()
        assert len(dirs) == len(set(dirs))


class TestResolution:
    def test_finds_a_library_in_a_probed_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(rnnoise.ctypes.util, "find_library", lambda _name: None)
        monkeypatch.setattr(rnnoise, "_rnnoise_search_dirs", lambda: [str(tmp_path)])
        soname = "librnnoise.dylib" if rnnoise.sys.platform == "darwin" else "librnnoise.so"
        (tmp_path / soname).write_bytes(b"")

        assert rnnoise._find_rnnoise() == str(tmp_path / soname)

    def test_returns_none_when_nothing_is_installed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        monkeypatch.setattr(rnnoise.ctypes.util, "find_library", lambda _name: None)
        monkeypatch.setattr(rnnoise, "_rnnoise_search_dirs", lambda: [str(tmp_path)])
        monkeypatch.setattr(rnnoise, "_pyrnnoise_library", lambda: None)
        assert rnnoise._find_rnnoise() is None


class TestPyrnnoiseWheel:
    """``roomkit[rnnoise]``: the library the pyrnnoise wheel ships."""

    @staticmethod
    def _fake_wheel(monkeypatch: pytest.MonkeyPatch, tmp_path) -> str:
        package = tmp_path / "site" / "pyrnnoise"
        package.mkdir(parents=True)
        # Importing this would fail the test: the loader must never import it.
        (package / "__init__.py").write_text("raise AssertionError('pyrnnoise was imported')\n")
        if rnnoise.sys.platform == "darwin":
            name = "librnnoise.dylib"
        elif rnnoise.sys.platform == "win32":
            name = "rnnoise.dll"
        else:
            name = "librnnoise.so"
        (package / name).write_bytes(b"")
        monkeypatch.syspath_prepend(str(tmp_path / "site"))
        return str(package / name)

    def test_found_when_the_system_has_none(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        library = self._fake_wheel(monkeypatch, tmp_path)
        monkeypatch.setattr(rnnoise.ctypes.util, "find_library", lambda _name: None)
        monkeypatch.setattr(rnnoise, "_rnnoise_search_dirs", lambda: [str(tmp_path / "none")])

        assert rnnoise._find_rnnoise() == library
        assert "pyrnnoise" not in rnnoise.sys.modules

    def test_the_system_library_comes_first(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        self._fake_wheel(monkeypatch, tmp_path)
        monkeypatch.setattr(rnnoise.ctypes.util, "find_library", lambda _name: "librnnoise.so.0")

        assert rnnoise._find_rnnoise() == "librnnoise.so.0"

    def test_none_without_the_wheel(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(rnnoise.importlib.util, "find_spec", lambda _name: None)
        assert rnnoise._pyrnnoise_library() is None


class TestGuidance:
    def test_error_does_not_send_macos_users_to_the_cask(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(rnnoise, "_lib", None)
        monkeypatch.setattr(rnnoise, "_find_rnnoise", lambda: None)

        with pytest.raises(ImportError) as excinfo:
            rnnoise._load_rnnoise()

        message = str(excinfo.value)
        assert "brew install rnnoise" not in message
        # A reader who cannot install it still has a way forward, and can see
        # where the loader actually looked.
        assert "SherpaOnnxDenoiserProvider" in message
        assert "Searched:" in message
