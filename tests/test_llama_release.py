"""Runtime packages: choice per machine, verified download, safe unpacking. No network."""

import hashlib
import http.server
import io
import re
import tarfile
import threading
import zipfile
from pathlib import Path

import pytest

from rizzo_flow import llama_release as release


def at(monkeypatch: pytest.MonkeyPatch, system: str, machine: str, nvidia: bool = False) -> None:
    monkeypatch.setattr(release, "host", lambda: (system, machine))
    monkeypatch.setattr(release, "nvidia_driver", lambda: nvidia)


@pytest.mark.parametrize(
    ("system", "machine", "nvidia", "expected"),
    [
        ("darwin", "arm64", False, "metal"),
        ("win32", "x64", True, "cuda"),
        ("win32", "x64", False, "vulkan"),  # AMD, Intel: one build for every GPU vendor
        ("linux", "x64", True, "cuda"),
        ("linux", "x64", False, "vulkan"),
        ("linux", "arm64", False, "vulkan"),
        ("win32", "arm64", False, "cpu"),
        ("darwin", "x64", False, "cpu"),
    ],
)
def test_auto_picks_the_package_for_the_machine(
    monkeypatch: pytest.MonkeyPatch, system: str, machine: str, nvidia: bool, expected: str
) -> None:
    at(monkeypatch, system, machine, nvidia)
    assert release.pick("auto") == expected


def test_explicit_family_must_exist_for_the_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    at(monkeypatch, "darwin", "arm64")
    with pytest.raises(ValueError, match="No cuda package"):
        release.pick("cuda")
    at(monkeypatch, "linux", "x64")
    assert release.pick("rocm") == "rocm"
    with pytest.raises(ValueError, match="one of"):
        release.pick("tpu")


def test_unknown_machine_is_told_how_to_bring_a_build(monkeypatch: pytest.MonkeyPatch) -> None:
    at(monkeypatch, "freebsd", "riscv64")
    with pytest.raises(ValueError, match=release.RUNTIME_DIR_ENV):
        release.pick("auto")


def test_every_package_is_pinned_by_a_sha256() -> None:
    for (system, machine, family), archives in release.PACKAGES.items():
        assert family in release.PREFERENCE, (system, machine, family)
        for asset in archives:
            assert re.fullmatch(r"[0-9a-f]{64}", asset.sha256), asset.name
            assert asset.name.endswith((".zip", ".tar.gz"))
            # Only the Windows CUDA runtime archive is shared between releases.
            assert release.RELEASE in asset.name or asset.name.startswith("cudart-llama-bin-win")


def archive_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        for name, data in members.items():
            bundle.writestr(name, data)


def archive_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w:gz") as bundle:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            bundle.addfile(info, io.BytesIO(data))


def test_tarballs_lose_their_wrapping_folder(tmp_path: Path) -> None:
    archive_tar(tmp_path / "a.tar.gz", {"llama-bX/libllama.so": b"1", "llama-bX/sub/x": b"2"})
    archive_tar(tmp_path / "b.tar.gz", {"cudart-bX/libcudart.so.13": b"3"})
    for name in ("a.tar.gz", "b.tar.gz"):
        release.unpack(tmp_path / name, tmp_path / "out")
    found = sorted(
        p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*")
    )
    assert found == ["libcudart.so.13", "libllama.so", "sub", "sub/x"]


def test_archives_cannot_write_outside_the_destination(tmp_path: Path) -> None:
    archive_zip(tmp_path / "bad.zip", {"../escaped.dll": b"x"})
    with pytest.raises(ValueError, match="unsafe"):
        release.unpack(tmp_path / "bad.zip", tmp_path / "out")
    archive_tar(tmp_path / "bad.tar.gz", {"top/../../escaped.so": b"x"})
    with pytest.raises(tarfile.FilterError):
        release.unpack(tmp_path / "bad.tar.gz", tmp_path / "out")
    assert not (tmp_path / "escaped.dll").exists() and not (tmp_path / "escaped.so").exists()


def test_fetch_verifies_and_never_keeps_a_bad_file(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"weights")
    good = hashlib.sha256(b"weights").hexdigest()
    target = tmp_path / "cache" / "file.bin"
    seen = []
    release.fetch(source.as_uri(), target, good, lambda *call: seen.append(call))
    assert target.read_bytes() == b"weights" and seen[-1][1] == 7
    with pytest.raises(ValueError, match="sha256 mismatch"):
        release.fetch(source.as_uri(), tmp_path / "other.bin", "0" * 64)
    assert not (tmp_path / "other.bin").exists() and not (tmp_path / "other.bin.part").exists()
    source.unlink()  # a verified copy is reused without touching the source
    assert release.fetch(source.as_uri(), target, good) == target


def test_interrupted_download_resumes_where_it_stopped(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 64
    requests = []

    class Flaky(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            requests.append(self.headers.get("Range"))
            if self.headers.get("Range"):
                start = int(self.headers["Range"].removeprefix("bytes=").rstrip("-"))
                self.send_response(206)
                body = payload[start:]
            else:
                self.send_response(200)
                body = payload
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            # The first answer promises everything and delivers a third, then hangs up.
            self.wfile.write(body[: len(body) // 3] if len(requests) == 1 else body)

        def log_message(self, *args: object) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Flaky)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/weights.gguf"
        target = release.fetch(url, tmp_path / "weights.gguf", hashlib.sha256(payload).hexdigest())
    finally:
        server.shutdown()
    assert target.read_bytes() == payload
    assert requests == [None, f"bytes={len(payload) // 3}-"]


def test_install_and_locate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    at(monkeypatch, "win32", "x64", nvidia=False)
    monkeypatch.setattr(release, "library_name", lambda: "llama.dll")
    monkeypatch.setattr(release, "RUNTIMES", tmp_path / "runtimes")
    monkeypatch.delenv(release.RUNTIME_DIR_ENV, raising=False)
    with pytest.raises(ValueError, match="rizzo download"):
        release.locate()
    served = tmp_path / "served"
    served.mkdir()
    packages = {}
    for family in ("vulkan", "cpu"):
        name = f"llama-{family}.zip"
        archive_zip(served / name, {"llama.dll": family.encode(), "ggml.dll": b"g"})
        packages[("win32", "x64", family)] = [
            release.Archive(name, release.sha256_file(served / name))
        ]
    monkeypatch.setattr(release, "PACKAGES", packages)
    monkeypatch.setattr(release, "BASE_URL", served.as_uri())
    directory = release.install("auto")
    assert directory == release.install_dir("vulkan") and (directory / "ggml.dll").is_file()
    assert release.install("auto") == directory  # idempotent
    release.install("cpu")
    assert release.installed() == ["vulkan", "cpu"]
    assert release.locate() == release.install_dir("vulkan")
    assert release.locate("cpu") == release.install_dir("cpu")
    assert release.locate("cuda") == release.install_dir("vulkan")  # not installed: best one
    own = tmp_path / "own" / "build" / "bin"
    own.mkdir(parents=True)
    (own / "llama.dll").write_bytes(b"x")
    monkeypatch.setenv(release.RUNTIME_DIR_ENV, str(own.parent))
    assert release.locate() == own  # one level of nesting is searched
    monkeypatch.setenv(release.RUNTIME_DIR_ENV, str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="not found"):
        release.locate()
