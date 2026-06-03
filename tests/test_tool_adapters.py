def test_competition_evaluate_path_intact():
    from edge_ran_gary.submission_adapter import get_evaluate_fn, load_submission_module, resolve_repo_root
    subs = resolve_repo_root() / "submissions"
    assert subs.is_dir()
    # At least one submission package exposes evaluate(filename)
    found = False
    for d in subs.iterdir():
        main_py = d / "main.py"
        if main_py.is_file():
            mod = load_submission_module(d)
            assert callable(get_evaluate_fn(mod))
            found = True
            break
    assert found, "no submission main.py with evaluate() found"

def test_oran_export(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from airan_research.tool_adapters.oran_kpi_schema import export
    p = export()
    assert p.exists()
