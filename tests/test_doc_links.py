"""The doc-link gate finds dead links and stays quiet about everything else."""

from pathlib import Path

from doc_links import BrokenLink, find_broken_links, iter_markdown_files, parse_links


def write(root: Path, name: str, body: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_finds_a_link_to_a_missing_file(tmp_path: Path) -> None:
    write(tmp_path, "index.md", "See [the contract](contract.md).")

    broken = find_broken_links(tmp_path)

    assert len(broken) == 1
    assert broken[0].source == Path("index.md")
    assert broken[0].line == 1
    assert broken[0].target == "contract.md"


def test_stays_quiet_when_the_target_exists(tmp_path: Path) -> None:
    write(tmp_path, "index.md", "See [the contract](contract.md).")
    write(tmp_path, "contract.md", "# Contract")

    assert find_broken_links(tmp_path) == []


def test_resolves_relative_to_the_containing_file_not_the_root(tmp_path: Path) -> None:
    write(tmp_path, "docs/adr/0001.md", "Back to [the index](../README.md).")
    write(tmp_path, "docs/README.md", "# Docs")

    assert find_broken_links(tmp_path) == []


def test_decodes_percent_encoded_spaces(tmp_path: Path) -> None:
    # Real repositories have directories with spaces in their names.
    write(tmp_path, "index.md", "Read [the notes](design%20notes/01_overview.md).")
    write(tmp_path, "design notes/01_overview.md", "# Overview")

    assert find_broken_links(tmp_path) == []


def test_ignores_the_anchor_when_resolving(tmp_path: Path) -> None:
    write(tmp_path, "index.md", "Jump to [a section](contract.md#why-it-is-shaped-this-way).")
    write(tmp_path, "contract.md", "# Contract")

    assert find_broken_links(tmp_path) == []


def test_ignores_external_and_same_file_links(tmp_path: Path) -> None:
    write(
        tmp_path,
        "index.md",
        "[site](https://example.com) [mail](mailto:a@b.c) [here](#section) [empty]()",
    )

    assert find_broken_links(tmp_path) == []


def test_ignores_links_inside_fenced_code_blocks(tmp_path: Path) -> None:
    write(
        tmp_path,
        "index.md",
        "Example:\n\n```markdown\n[not a real link](nowhere.md)\n```\n\nDone.\n",
    )

    assert find_broken_links(tmp_path) == []


def test_skips_vendor_and_build_directories(tmp_path: Path) -> None:
    write(tmp_path, "node_modules/pkg/README.md", "[dead](nowhere.md)")
    write(tmp_path, ".venv/lib/NOTES.md", "[dead](nowhere.md)")

    assert find_broken_links(tmp_path) == []
    assert iter_markdown_files(tmp_path) == []


def test_reports_every_broken_link_on_a_line(tmp_path: Path) -> None:
    write(tmp_path, "index.md", "[a](a.md) and [b](b.md)")

    broken = find_broken_links(tmp_path)

    assert [link.target for link in broken] == ["a.md", "b.md"]
    assert {link.line for link in broken} == {1}


def test_parse_links_reports_one_based_line_numbers() -> None:
    text = "# Title\n\nSee [the thing](thing.md).\n"

    assert parse_links(text) == [(3, "thing.md")]


def test_broken_link_renders_as_an_editor_jump_target() -> None:
    link = BrokenLink(
        source=Path("docs/entry.md"), line=12, target="gone.md", resolved=Path("docs/gone.md")
    )

    assert str(link) == "docs/entry.md:12: gone.md -> docs/gone.md"
