# SPDX-License-Identifier: Apache-2.0
"""Pin the vendored NVIDIA CUPTI binaries to their provenance docs.

Checks ``agent/collection_framework/src/third_party/cupti``:

* every row of the ``README.md`` markdown table (File / Bytes / SHA-256 /
  ELF SONAME / CUDA Toolkit line) matches the binary on disk — size,
  ``hashlib.sha256`` digest, and the ``DT_SONAME`` read out of the ELF
  image with a small pure-python ELF64 parser (no ``readelf``);
* every row of the ``NOTICE`` fixed-width table (file / CUPTI release /
  ELF SONAME / CUDA generation) matches the binary's ``DT_SONAME``;
* the two tables agree on the file set, on every SONAME and on the CUDA
  generation, and no ``libcupti.so.*`` on disk is undocumented;
* the README prose total ("Ten files, N bytes in total") equals both the
  sum of the documented per-file sizes and the sum of the real sizes.

Run with ``python3.11 -m pytest test/unit/test_cupti_manifest.py -v``.
"""

from __future__ import annotations

import hashlib
import re
import struct
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CUPTI_DIR = (
    ROOT / "agent" / "collection_framework" / "src" / "third_party" / "cupti"
)

# ELF64 little-endian constants used by _elf_soname().
_ELF_MAGIC = b"\x7fELF"
_ELFCLASS64 = 2
_ELFDATA2LSB = 1
_PT_LOAD = 1
_PT_DYNAMIC = 2
_DT_NULL = 0
_DT_STRTAB = 5
_DT_SONAME = 14


def _require_cupti_dir() -> Path:
    """Skip (not fail) when the vendored tree or its binaries are absent."""
    if not CUPTI_DIR.is_dir():
        pytest.skip(f"vendored CUPTI directory not present: {CUPTI_DIR}")
    if not sorted(CUPTI_DIR.glob("libcupti.so.*")):
        pytest.skip(
            f"no libcupti.so.* binaries in {CUPTI_DIR} (partial checkout)"
        )
    return CUPTI_DIR


def _require_doc(directory: Path, name: str) -> Path:
    path = directory / name
    if not path.is_file():
        pytest.skip(f"{name} not present in {directory} (partial checkout)")
    return path


def _parse_readme_table(path: Path) -> list[dict[str, str]]:
    """Parse the README markdown table into row dicts.

    Strategy: keep only lines whose first non-space character is '|', drop
    the header row and the '|---|' separator row, split the rest on '|',
    then strip whitespace and the backtick quoting used for the File,
    SHA-256 and SONAME cells.
    """
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        if not cells or not cells[0]:
            continue
        if set(cells[0]) <= {"-", ":", " "}:  # the |----|----| separator row
            continue
        if cells[0].lower() == "file":  # the header row
            continue
        if len(cells) < 5:
            raise AssertionError(f"malformed README table row: {line!r}")
        rows.append(
            {
                "file": cells[0],
                "bytes": cells[1],
                "sha256": cells[2],
                "soname": cells[3],
                "cuda": cells[4],
            }
        )
    if not rows:
        raise AssertionError(f"no markdown table rows found in {path}")
    return rows


def _parse_readme_total(path: Path) -> int:
    """Extract the byte total from the README prose sentence."""
    text = path.read_text(encoding="utf-8")
    match = re.search(r"([\d][\d,]*)\s+bytes in total", text)
    if match is None:
        raise AssertionError(f"no '... bytes in total' sentence in {path}")
    return int(match.group(1).replace(",", ""))


def _parse_notice_table(path: Path) -> list[dict[str, str]]:
    """Parse the NOTICE fixed-width table into row dicts.

    Strategy: the table is the indented block under "Binaries as shipped".
    A data row is any line whose first whitespace-separated token matches
    ``libcupti.so.<digits-and-dots>`` (which also excludes the column
    header). Columns are whitespace separated, so the trailing two-word
    CUDA generation is simply the remainder of the line.
    """
    rows: list[dict[str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 4 or not parts[0].startswith("libcupti.so."):
            continue
        if not re.fullmatch(r"libcupti\.so\.[0-9][0-9.]*", parts[0]):
            continue
        rows.append(
            {
                "file": parts[0],
                "release": parts[1],
                "soname": parts[2],
                "cuda": " ".join(parts[3:]),
            }
        )
    if not rows:
        raise AssertionError(f"no fixed-width table rows found in {path}")
    return rows


def _elf_soname(blob: bytes) -> str:
    """Read DT_SONAME out of an in-memory ELF64 little-endian image.

    ELF header -> program headers -> PT_DYNAMIC -> dynamic entries
    (DT_STRTAB, DT_SONAME). DT_STRTAB holds a virtual address, so it is
    translated back to a file offset via the PT_LOAD segment covering it
    before the NUL-terminated string is read.
    """
    if blob[:4] != _ELF_MAGIC:
        raise AssertionError("not an ELF file (bad magic)")
    if blob[4] != _ELFCLASS64:
        raise AssertionError("not an ELF64 file")
    if blob[5] != _ELFDATA2LSB:
        raise AssertionError("not little-endian ELF data")

    (e_phoff,) = struct.unpack_from("<Q", blob, 32)
    e_phentsize, e_phnum = struct.unpack_from("<HH", blob, 54)

    loads: list[tuple[int, int, int]] = []  # (p_vaddr, p_offset, p_filesz)
    dynamic: tuple[int, int] | None = None  # (p_offset, p_filesz)
    for i in range(e_phnum):
        base = e_phoff + i * e_phentsize
        (p_type,) = struct.unpack_from("<I", blob, base)
        p_offset, p_vaddr, _p_paddr, p_filesz, _p_memsz, _p_align = (
            struct.unpack_from("<QQQQQQ", blob, base + 8)
        )
        if p_type == _PT_LOAD:
            loads.append((p_vaddr, p_offset, p_filesz))
        elif p_type == _PT_DYNAMIC and dynamic is None:
            dynamic = (p_offset, p_filesz)
    if dynamic is None:
        raise AssertionError("no PT_DYNAMIC program header")

    dyn_off, dyn_size = dynamic
    strtab_vaddr: int | None = None
    soname_str_off: int | None = None
    for off in range(dyn_off, dyn_off + dyn_size - 15, 16):
        d_tag, d_val = struct.unpack_from("<qQ", blob, off)
        if d_tag == _DT_NULL:
            break
        if d_tag == _DT_STRTAB:
            strtab_vaddr = d_val
        elif d_tag == _DT_SONAME:
            soname_str_off = d_val
    if strtab_vaddr is None:
        raise AssertionError("no DT_STRTAB entry in the dynamic section")
    if soname_str_off is None:
        raise AssertionError("no DT_SONAME entry in the dynamic section")

    strtab_off = None
    for p_vaddr, p_offset, p_filesz in loads:
        if p_vaddr <= strtab_vaddr < p_vaddr + p_filesz:
            strtab_off = p_offset + (strtab_vaddr - p_vaddr)
            break
    if strtab_off is None:
        raise AssertionError(
            f"DT_STRTAB vaddr 0x{strtab_vaddr:x} is not inside any PT_LOAD"
        )

    start = strtab_off + soname_str_off
    end = blob.index(b"\x00", start)
    return blob[start:end].decode("ascii")


def _read_once(directory: Path, name: str) -> tuple[bytes, int]:
    """Read a binary exactly once; return (contents, stat size)."""
    path = directory / name
    size = path.stat().st_size
    return path.read_bytes(), size


def _report(problems: list[str]) -> None:
    if problems:
        raise AssertionError(
            f"{len(problems)} manifest mismatch(es):\n  "
            + "\n  ".join(problems)
        )


def test_readme_table_matches_binaries() -> None:
    """README rows: file present, size, SHA-256 and ELF SONAME all match."""
    directory = _require_cupti_dir()
    readme = _require_doc(directory, "README.md")
    problems: list[str] = []
    for row in _parse_readme_table(readme):
        name = row["file"]
        if not (directory / name).is_file():
            problems.append(f"{name}: documented in README.md, absent on disk")
            continue
        blob, size = _read_once(directory, name)
        if size != int(row["bytes"]):
            problems.append(f"{name}: size {size} != documented {row['bytes']}")
        digest = hashlib.sha256(blob).hexdigest()
        if digest != row["sha256"]:
            problems.append(f"{name}: sha256 {digest} != {row['sha256']}")
        soname = _elf_soname(blob)
        if soname != row["soname"]:
            problems.append(
                f"{name}: DT_SONAME {soname!r} != documented {row['soname']!r}"
            )
    _report(problems)


def test_notice_table_matches_binaries() -> None:
    """NOTICE rows: file present, CUPTI release and ELF SONAME match."""
    directory = _require_cupti_dir()
    notice = _require_doc(directory, "NOTICE")
    problems: list[str] = []
    for row in _parse_notice_table(notice):
        name = row["file"]
        if not (directory / name).is_file():
            problems.append(f"{name}: documented in NOTICE, absent on disk")
            continue
        blob, _size = _read_once(directory, name)
        soname = _elf_soname(blob)
        if soname != row["soname"]:
            problems.append(
                f"{name}: DT_SONAME {soname!r} != NOTICE {row['soname']!r}"
            )
        if row["release"] not in name:
            problems.append(
                f"{name}: NOTICE release {row['release']!r} not in file name"
            )
    _report(problems)


def test_readme_and_notice_tables_agree() -> None:
    """Both docs name the same files with the same SONAME / CUDA line."""
    directory = _require_cupti_dir()
    readme_rows = _parse_readme_table(_require_doc(directory, "README.md"))
    notice_rows = _parse_notice_table(_require_doc(directory, "NOTICE"))
    readme_by_name = {r["file"]: r for r in readme_rows}
    notice_by_name = {r["file"]: r for r in notice_rows}

    assert len(readme_by_name) == len(readme_rows), "duplicate README rows"
    assert len(notice_by_name) == len(notice_rows), "duplicate NOTICE rows"
    assert set(readme_by_name) == set(notice_by_name), (
        "README/NOTICE file sets differ: "
        f"README only={sorted(set(readme_by_name) - set(notice_by_name))}, "
        f"NOTICE only={sorted(set(notice_by_name) - set(readme_by_name))}"
    )
    for name, readme in sorted(readme_by_name.items()):
        notice = notice_by_name[name]
        assert readme["soname"] == notice["soname"], (
            f"{name}: SONAME README={readme['soname']!r} "
            f"NOTICE={notice['soname']!r}"
        )
        assert readme["cuda"] == notice["cuda"], (
            f"{name}: CUDA line README={readme['cuda']!r} "
            f"NOTICE={notice['cuda']!r}"
        )


def test_every_binary_on_disk_is_documented() -> None:
    """A libcupti.so.* on disk that neither table lists is a failure."""
    directory = _require_cupti_dir()
    readme_names = {
        r["file"]
        for r in _parse_readme_table(_require_doc(directory, "README.md"))
    }
    notice_names = {
        r["file"] for r in _parse_notice_table(_require_doc(directory, "NOTICE"))
    }
    on_disk = {p.name for p in directory.glob("libcupti.so.*") if p.is_file()}
    assert on_disk - readme_names == set(), (
        f"not listed in README.md: {sorted(on_disk - readme_names)}"
    )
    assert on_disk - notice_names == set(), (
        f"not listed in NOTICE: {sorted(on_disk - notice_names)}"
    )


def test_documented_byte_totals() -> None:
    """README prose total == sum of documented sizes == sum of real sizes."""
    directory = _require_cupti_dir()
    readme = _require_doc(directory, "README.md")
    rows = _parse_readme_table(readme)
    stated_total = _parse_readme_total(readme)

    documented = 0
    actual = 0
    problems: list[str] = []
    for row in rows:
        name = row["file"]
        documented += int(row["bytes"])
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: documented in README.md, absent on disk")
            continue
        size = path.stat().st_size
        actual += size
        if size != int(row["bytes"]):
            problems.append(f"{name}: size {size} != documented {row['bytes']}")
    _report(problems)

    assert stated_total == documented, (
        f"README prose total {stated_total} != sum of documented sizes "
        f"{documented}"
    )
    assert stated_total == actual, (
        f"README prose total {stated_total} != sum of actual sizes {actual}"
    )
