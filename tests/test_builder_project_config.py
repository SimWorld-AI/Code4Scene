"""The dataset builder writes the package redirects and renderer settings into the project."""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:  # dataset_builder is run from the repository root, not installed
    sys.path.insert(0, str(REPO))

from dataset_builder import build  # noqa: E402


def _wanted():
    lines = build.PACKAGE_REDIRECTS.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _project(tmp_path, ini_text=None):
    project = tmp_path / "Proj" / "Proj.uproject"
    project.parent.mkdir(parents=True)
    project.write_text('{"FileVersion": 3, "Plugins": []}\n')
    if ini_text is not None:
        (project.parent / "Config").mkdir()
        (project.parent / "Config" / "DefaultEngine.ini").write_text(ini_text)
    return project


def test_redirect_lines_are_well_formed():
    wanted = _wanted()
    assert wanted
    for line in wanted:
        assert line.startswith('+PackageRedirects=(OldName="/Game/') and '",NewName="/Game/' in line


def test_redirects_are_added_once_and_existing_settings_are_kept(tmp_path):
    existing = "[/Script/EngineSettings.GameMapsSettings]\nGameDefaultMap=/Game/Maps/Start\n"
    project = _project(tmp_path, existing)
    ini = project.parent / "Config" / "DefaultEngine.ini"
    assert build.ensure_package_redirects(project) == len(_wanted())
    text = ini.read_text()
    assert text.startswith(existing) and "[CoreRedirects]" in text
    assert all(line in text.splitlines() for line in _wanted())
    assert (ini.parent / "DefaultEngine.ini.bak").read_text() == existing
    assert build.ensure_package_redirects(project) == 0
    assert ini.read_text() == text


def test_redirects_join_an_existing_core_redirects_section(tmp_path):
    project = _project(tmp_path, "[CoreRedirects]\n+ClassRedirects=(OldName=\"A\",NewName=\"B\")\n")
    build.ensure_package_redirects(project)
    lines = (project.parent / "Config" / "DefaultEngine.ini").read_text().splitlines()
    assert lines.count("[CoreRedirects]") == 1 and lines[0] == "[CoreRedirects]"
    assert '+ClassRedirects=(OldName="A",NewName="B")' in lines


def test_a_project_without_config_gets_one_and_dry_run_writes_nothing(tmp_path):
    project = _project(tmp_path)
    assert build.ensure_package_redirects(project, dry_run=True) == len(_wanted())
    assert not (project.parent / "Config").exists()
    build.ensure_package_redirects(project)
    assert (project.parent / "Config" / "DefaultEngine.ini").is_file()


def _settings():
    lines = build.RENDERER_SETTINGS.read_text(encoding="utf-8").splitlines()
    return dict(line.strip().split("=", 1) for line in lines if line.strip() and not line.startswith("#"))


def _section(ini):
    lines = ini.read_text().splitlines()
    start = lines.index(build.RENDERER_SECTION)
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("[")), len(lines))
    return [line for line in lines[start + 1:end] if line.strip()]


def test_renderer_settings_replace_other_values_and_keep_everything_else(tmp_path):
    existing = ("[/Script/Engine.RendererSettings]\nr.DynamicGlobalIlluminationMethod=0\nr.CustomDepth=3\n\n"
                "[/Script/EngineSettings.GameMapsSettings]\nGameDefaultMap=/Game/Maps/Start\n")
    project = _project(tmp_path, existing)
    ini = project.parent / "Config" / "DefaultEngine.ini"
    assert build.ensure_renderer_settings(project) == len(_settings())
    section = dict(line.split("=", 1) for line in _section(ini))
    assert section == {**_settings(), "r.CustomDepth": "3"}
    assert "GameDefaultMap=/Game/Maps/Start" in ini.read_text().splitlines()
    assert build.ensure_renderer_settings(project) == 0


def test_init_project_keeps_the_original_ini_as_the_backup(tmp_path):
    existing = "[/Script/EngineSettings.GameMapsSettings]\nGameDefaultMap=/Game/Maps/Start\n"
    project = _project(tmp_path, existing)
    build.ensure_package_redirects(project)
    build.ensure_renderer_settings(project)
    ini = project.parent / "Config" / "DefaultEngine.ini"
    assert (ini.parent / "DefaultEngine.ini.bak").read_text() == existing
    assert dict(line.split("=", 1) for line in _section(ini)) == _settings()


def test_renderer_settings_dry_run_writes_nothing(tmp_path):
    project = _project(tmp_path)
    assert build.ensure_renderer_settings(project, dry_run=True) == len(_settings())
    assert not (project.parent / "Config").exists()

