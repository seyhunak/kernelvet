"""TUI startup, navigation and workflow tests using Textual's pilot (headless)."""

from __future__ import annotations

from pathlib import Path

from textual.widgets import Button, Checkbox, ContentSwitcher, DataTable, Input

from kernelvet.config import Settings, config_path, load_settings
from kernelvet.history import HistoryStore
from kernelvet.tui.app import SECTIONS, KernelVetApp
from kernelvet.tui.screens.help import HelpScreen
from kernelvet.utils.proc import HostContext


def make_app(ctx: HostContext, tmp_path: Path) -> KernelVetApp:
    return KernelVetApp(ctx=ctx, settings=Settings(), history=HistoryStore(tmp_path / "hist"))


async def wait_workers(app: KernelVetApp, pilot) -> None:  # type: ignore[no-untyped-def]
    await app.workers.wait_for_complete()
    await pilot.pause()


async def test_startup_and_keyboard_navigation(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    app = make_app(mi300x_ctx, tmp_path)
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_workers(app, pilot)
        switcher = app.query_one("#content", ContentSwitcher)
        assert switcher.current == "overview"
        for i, (sid, _label, _cls) in enumerate(SECTIONS):
            await pilot.press(str(i + 1))
            assert switcher.current == sid
        # overview inventory was collected in a worker
        assert app.state.gpus and app.state.gpus[0].gfx_target == "gfx942"
        assert app.state.compatibility is not None


async def test_context_help(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    app = make_app(mi300x_ctx, tmp_path)
    async with app.run_test() as pilot:
        await pilot.press("2")
        await pilot.press("question_mark")
        assert isinstance(app.screen, HelpScreen)
        assert "Diagnose" in app.screen.text
        await pilot.press("escape")
        assert not isinstance(app.screen, HelpScreen)


async def test_resize(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    app = make_app(mi300x_ctx, tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_workers(app, pilot)
        await pilot.resize_terminal(200, 60)
        await pilot.pause()
        assert app.size.width == 200


async def test_diagnose_runs_in_worker_and_exports(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    app = make_app(mi300x_ctx, tmp_path)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("2")
        pane = app.pane("diagnose")
        pane.focus()
        await pilot.press("r")
        await wait_workers(app, pilot)
        assert app.state.diagnosis is not None and app.state.diagnosis.status.value == "HEALTHY"
        table = app.query_one("#dg-checks", DataTable)
        assert table.row_count == len(app.state.diagnosis.all_checks())
        assert not app.query_one("#dg-json", Button).disabled
        pane.action_export("markdown")
        exported = list(app.state.history.reports_dir.glob("diagnosis-*.md"))
        assert exported and "HEALTHY" in exported[0].read_text()
        assert app.state.history.recent_runs(kind="diagnosis")


async def test_migration_scan_plan_and_unavailable_convert(
    cuda_project: Path, empty_linux_ctx: HostContext, tmp_path: Path
) -> None:
    app = make_app(empty_linux_ctx, tmp_path)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.press("5")
        inp = app.query_one("#mg-project", Input)
        inp.value = str(cuda_project)
        app.query_one("#mg-scan", Button).press()
        await wait_workers(app, pilot)
        m = app.state.migration
        assert m.analysis is not None and m.analysis.readiness.score <= 60
        assert app.query_one("#mg-findings", DataTable).row_count == len(m.analysis.findings)
        assert app.query_one("#mg-output", Input).value.endswith("cuda-project-hip")
        # HIPIFY is not installed on this host: Convert is disabled and the reason is shown
        assert app.query_one("#mg-convert", Button).disabled
        assert "Convert unavailable" in str(app.query_one("#mg-availability").render())
        app.query_one("#mg-plan", Button).press()
        await pilot.pause()
        assert m.plan is not None and m.plan.translator is None


async def test_migration_full_flow_with_fake_tools(cuda_project: Path, tmp_path: Path) -> None:
    from conftest import fake_hipify

    runner = fake_hipify()
    ctx = HostContext(runner=runner, root=tmp_path / "sys", system="Linux")
    (tmp_path / "sys").mkdir()
    app = make_app(ctx, tmp_path)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.press("5")
        app.query_one("#mg-project", Input).value = str(cuda_project)
        app.query_one("#mg-scan", Button).press()
        await wait_workers(app, pilot)
        out = tmp_path / "out"
        app.query_one("#mg-output", Input).value = str(out)
        await pilot.pause()
        assert not app.query_one("#mg-convert", Button).disabled
        app.query_one("#mg-convert", Button).press()
        await wait_workers(app, pilot)
        assert app.state.migration.conversion is not None and (out / "src/vector_add.cu").exists()
        app.query_one("#mg-audit", Button).press()
        await wait_workers(app, pilot)
        assert app.state.migration.audit is not None and app.state.migration.audit.findings
        assert app.query_one("#mg-build", Button).disabled  # no hipcc on this host
        app.query_one("#mg-report", Button).press()
        await pilot.pause()
        names = {p.suffix for p in app.state.history.reports_dir.glob("migration-*")}
        assert names == {".json", ".md", ".html", ".sarif"}


async def test_settings_save(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    app = make_app(mi300x_ctx, tmp_path)
    async with app.run_test(size=(140, 50)) as pilot:
        await pilot.press("8")
        app.query_one("#st-command_timeout", Input).value = "42"
        app.query_one("#st-save", Button).press()
        await pilot.pause()
        assert config_path().exists()
        assert load_settings().command_timeout == 42
        app.query_one("#st-command_timeout", Input).value = "-1"
        app.query_one("#st-save", Button).press()
        await pilot.pause()
        assert load_settings().command_timeout == 42  # invalid value rejected


async def test_reports_pane_lists_saved_reports(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "hist")
    from kernelvet.diagnose.engine import DiagnoseOptions, run_diagnosis

    store.record_run("diagnosis", "HEALTHY", "ok", run_diagnosis(mi300x_ctx, DiagnoseOptions(torch=False)))
    app = KernelVetApp(ctx=mi300x_ctx, settings=Settings(), history=store)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("7")
        await pilot.pause()
        table = app.query_one("#rp-table", DataTable)
        assert table.row_count == 1
        table.focus()
        await pilot.press("down")
        await pilot.pause()
        app.query_one("#rp-md", Button).press()
        await pilot.pause()
        assert list(store.reports_dir.glob("*.md"))


async def test_benchmarks_skip_without_gpu(empty_linux_ctx: HostContext, tmp_path: Path) -> None:
    app = make_app(empty_linux_ctx, tmp_path)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("6")
        app.query_one("#bm-run", Button).press()
        await wait_workers(app, pilot)
        report = app.state.benchmarks
        assert report is not None and {r.status for r in report.results} == {"SKIPPED"}


def _pane_of(app: KernelVetApp) -> str | None:
    """Id of the section that holds keyboard focus (None if focus is outside every section)."""
    from kernelvet.tui.screens.base import Pane

    focused = app.focused
    if focused is None:
        return None
    return next((n.id for n in focused.ancestors_with_self if isinstance(n, Pane)), None)


async def test_number_key_hands_the_section_the_keyboard(
    empty_linux_ctx: HostContext, tmp_path: Path
) -> None:
    app = make_app(empty_linux_ctx, tmp_path)
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_workers(app, pilot)
        await pilot.press("2")
        assert _pane_of(app) == "diagnose"
        # what the Overview and Diagnose screens tell the user to do: "press 2 then r"
        await pilot.press("r")
        await wait_workers(app, pilot)
        assert app.state.diagnosis is not None


async def test_focus_never_stays_in_a_hidden_section(empty_linux_ctx: HostContext, tmp_path: Path) -> None:
    app = make_app(empty_linux_ctx, tmp_path)
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_workers(app, pilot)
        await pilot.press("6")
        app.query_one("#bm-run", Button).focus()  # e.g. after clicking Run benchmarks
        await pilot.pause()
        app.show_pane("reports")  # same path as moving through the sidebar
        await pilot.pause()
        assert _pane_of(app) == "reports"
        # the footer and keys belong to Reports now, not to Benchmarks' "r Run benchmarks"
        assert "r" not in app.active_bindings
        assert "ctrl+r" in app.active_bindings


async def test_button_bars_fit_small_terminals(empty_linux_ctx: HostContext, tmp_path: Path) -> None:
    from rich.cells import cell_len

    for size in ((80, 24), (100, 30)):
        app = make_app(empty_linux_ctx, tmp_path)
        async with app.run_test(size=size) as pilot:
            await wait_workers(app, pilot)
            for i, (sid, _label, _cls) in enumerate(SECTIONS):
                await pilot.press(str(i + 1))
                await pilot.pause()
                for bar in app.query(f"#{sid} .buttons"):
                    for child in bar.children:
                        r, b = child.region, bar.region
                        assert r.width > 0 and r.x >= b.x and r.right <= b.right, (size, sid, child.id)
                        if isinstance(child, Button):
                            assert child.content_region.width >= cell_len(str(child.label)), (size, child.id)
                        if isinstance(child, Checkbox):  # the label sits after a 4-cell "▐X▌ " box
                            assert child.content_region.width >= cell_len(str(child.label)) + 4, (
                                size,
                                child.id,
                            )


async def test_corrupt_history_shows_a_notice_not_a_crash(
    empty_linux_ctx: HostContext, tmp_path: Path
) -> None:
    store = HistoryStore(tmp_path / "hist")
    store.root.mkdir(parents=True)
    store.db_path.write_text("not a database")
    app = KernelVetApp(ctx=empty_linux_ctx, settings=Settings(), history=store)
    async with app.run_test(size=(140, 45)) as pilot:
        await wait_workers(app, pilot)
        await pilot.press("2")
        await pilot.press("r")
        await wait_workers(app, pilot)
        assert app.is_running and app.state.diagnosis is not None
        titles = [n.title for n in app._notifications]
        assert titles.count("History unavailable") == 1 and "History not saved" in titles


async def test_reports_pane_skips_files_that_are_not_reports(
    empty_linux_ctx: HostContext, tmp_path: Path
) -> None:
    store = HistoryStore(tmp_path / "hist")
    store.reports_dir.mkdir(parents=True)
    (store.reports_dir / "x-diagnosis.json").write_text("[]")
    app = KernelVetApp(ctx=empty_linux_ctx, settings=Settings(), history=store)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("7")
        await pilot.pause()
        app.query_one("#rp-table", DataTable).focus()
        await pilot.press("down")
        await pilot.pause()
        assert app.is_running


async def test_reports_keep_the_selected_report_after_saving(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    from kernelvet.diagnose.engine import DiagnoseOptions, run_diagnosis

    store = HistoryStore(tmp_path / "hist")
    report = run_diagnosis(mi300x_ctx, DiagnoseOptions(torch=False))
    first = store.record_run("diagnosis", "HEALTHY", "first", report)
    store.record_run("diagnosis", "HEALTHY", "second", report)  # newer: sorts above the first
    app = KernelVetApp(ctx=mi300x_ctx, settings=Settings(), history=store)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("7")
        await pilot.pause()
        table = app.query_one("#rp-table", DataTable)
        chosen = Path(str(first.report_path))
        table.focus()
        table.move_cursor(row=[str(r.value) for r in table.rows].index(str(chosen)))
        await pilot.pause()
        for button in ("#rp-md", "#rp-html"):  # the Markdown save refreshes the list in between
            app.query_one(button, Button).press()
            await pilot.pause(0.3)
        assert chosen.with_suffix(".md").exists() and chosen.with_suffix(".html").exists()
        assert not list(store.reports_dir.glob("*.html"))[1:]  # nothing converted from the other report


async def test_overview_panels_stack_on_an_80_column_terminal(
    empty_linux_ctx: HostContext, tmp_path: Path
) -> None:
    app = make_app(empty_linux_ctx, tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await wait_workers(app, pilot)
        env, health = app.query_one("#ov-env").region, app.query_one("#ov-health").region
        assert health.y > env.bottom - 1 and env.width > 40  # one above the other, full width
    async with make_app(empty_linux_ctx, tmp_path).run_test(size=(140, 45)) as pilot:
        await wait_workers(pilot.app, pilot)  # type: ignore[arg-type]
        env, health = pilot.app.query_one("#ov-env").region, pilot.app.query_one("#ov-health").region
        assert health.y == env.y  # side by side when there is room


async def test_settings_save_is_reachable_without_scrolling(mi300x_ctx: HostContext, tmp_path: Path) -> None:
    app = make_app(mi300x_ctx, tmp_path)
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.press("8")
        await pilot.pause()
        save = app.query_one("#st-save", Button).region
        assert save.height > 0 and save.bottom <= app.size.height
        field = app.query_one("#st-command_timeout", Input)
        field.focus()
        field.value = "41"
        await pilot.press("ctrl+s")  # works while typing in a field
        await pilot.pause()
        assert load_settings().command_timeout == 41
        field.value = "77"
        await pilot.press("ctrl+r")
        await pilot.pause()
        assert field.value == "41.0" and load_settings().command_timeout == 41


async def test_benchmarks_reject_iterations_the_cli_rejects(
    empty_linux_ctx: HostContext, tmp_path: Path
) -> None:
    app = make_app(empty_linux_ctx, tmp_path)
    async with app.run_test(size=(140, 45)) as pilot:
        await pilot.press("6")
        for value in ("1", "", "abc", "1001"):
            app.query_one("#bm-iters", Input).value = value
            seen = len(app._notifications)
            app.query_one("#bm-run", Button).press()
            await wait_workers(app, pilot)
            assert app.state.benchmarks is None, value
            assert "iterations must be a whole number from 3 to 1000" in str(
                list(app._notifications)[seen:][0].message
            )
        app.query_one("#bm-iters", Input).value = "5"
        app.query_one("#bm-run", Button).press()
        await wait_workers(app, pilot)
        assert app.state.benchmarks is not None
