"""Update only numeric LaTeX data from the RAM retrieval Excel workbook.

python STORM-1/results/update_tex_ram_retrieval.py
python STORM-1/results/update_tex_ram_retrieval.py --refresh-excel

The manuscript is read only. Values are written to ram_retrieval_numbers.tex,
which the manuscript inputs. All wording and table formatting stay in the
manuscript. --refresh-excel first runs export_statistics.py.
"""
import argparse
from dataclasses import dataclass
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEX = ROOT.parent / "iclr2027_conference.tex"
DEFAULT_EXCEL = ROOT / "results/ram_verification/retrieval_statistics.xlsx"
GAMES = ("pong", "seaquest", "frostbite")
NAMES = dict(pong="Pong", seaquest="Seaquest", frostbite="Frostbite")
# The appendix describes precisely this distance definition and RAM decoding.
# Change the method description before accepting data from another definition.
METRIC_SHA256 = "38c0feed47d375298a552ec7191c832b192a672a40f8687a37f35d84f4ffd7f4"
RAM_SHA256 = "30df48c90130ffa7def7f911d67bf5a7b4dfe321f00cfb2ff621ba496f896d48"
NUMBERS_FILENAME = "ram_retrieval_numbers.tex"
PREFIXES = dict(pong="Pong", seaquest="Seaquest", frostbite="Frostbite")


@dataclass
class GameReport:
    game: str
    summary: dict
    metrics: dict
    identity: dict
    seeds: list


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Missing or invalid numeric field: {name}")
    return float(value)


def same_number(actual, expected, name):
    if actual is None and expected is None:
        return
    if not math.isclose(finite(actual, name), finite(expected, name), rel_tol=1e-8, abs_tol=1e-11):
        raise ValueError(f"Workbook sheets disagree: {name}")


def read_sheet(workbook, name):
    if name not in workbook.sheetnames:
        raise ValueError(f"Missing worksheet {name}; regenerate the statistics workbook")
    iterator = iter(workbook[name].values)
    headers = next(iterator, ())
    if not headers or len(set(headers)) != len(headers):
        raise ValueError(f"Invalid worksheet header: {name}")
    return [dict(zip(headers, row)) for row in iterator if any(v is not None for v in row)]


def load_reports(excel, selections=None):
    selections = selections or {}
    workbook = load_workbook(excel, read_only=True, data_only=True)
    try:
        summaries = read_sheet(workbook, "Summary")
        metrics = read_sheet(workbook, "Statistics")
        seeds = read_sheet(workbook, "Seeds")
        groups = read_sheet(workbook, "Groups")
    finally:
        workbook.close()
    reports = []
    for game in GAMES:
        candidates = [r for r in summaries if r.get("game") == game]
        if game in selections:
            candidates = [r for r in candidates if r.get("group_id") == selections[game]]
        if len(candidates) != 1:
            ids = [r.get("group_id") for r in summaries if r.get("game") == game]
            raise ValueError(f"Expected one {game} group, found {ids}. Select with --group {game}=GROUP_ID")
        summary = candidates[0]
        group_id = summary["group_id"]
        identities = [r for r in groups if r.get("group_id") == group_id]
        if len(identities) != 1:
            raise ValueError(f"Expected one protocol identity for {group_id}")
        identity = json.loads(identities[0]["protocol_identity_json"])
        if identity.get("game") != game or identity.get("metric_sha256") != METRIC_SHA256:
            raise ValueError(f"{group_id}: metric differs from the documented appendix definition")
        seed_rows = sorted([r for r in seeds if r.get("group_id") == group_id], key=lambda r: r["evaluation_seed"])
        n = int(finite(summary.get("n_seeds"), "n_seeds"))
        if n < 1 or len(seed_rows) != n or len({r["evaluation_seed"] for r in seed_rows}) != n:
            raise ValueError(f"{group_id}: inconsistent seed count")
        metric_rows = [r for r in metrics if r.get("group_id") == group_id]
        by_metric = {r.get("metric"): r for r in metric_rows}
        if len(metric_rows) != 3 or set(by_metric) != {"retrieval_similarity", "random_similarity", "similarity_gain"}:
            raise ValueError(f"{group_id}: incomplete Statistics rows")
        for metric, summary_mean, summary_sd in (
                ("retrieval_similarity", "retrieval_mean", "retrieval_seed_sd"),
                ("random_similarity", "random_mean", "random_seed_sd"),
                ("similarity_gain", "similarity_gain", "gain_seed_sd")):
            row = by_metric[metric]
            values = [finite(r.get(metric + "_mean"), metric) for r in seed_rows]
            expected_sd = statistics.stdev(values) if n > 1 else None
            same_number(summary.get(summary_mean), statistics.mean(values), summary_mean)
            same_number(summary.get(summary_sd), expected_sd, summary_sd)
            same_number(row.get("seed_equal_mean"), summary.get(summary_mean), metric + " mean")
            same_number(row.get("seed_sample_sd"), summary.get(summary_sd), metric + " SD")
        same_number(summary.get("similarity_gain"), summary["retrieval_mean"] - summary["random_mean"], "similarity gain")
        if sum(int(r["paired_queries"]) for r in seed_rows) != int(summary["paired_queries"]):
            raise ValueError(f"{group_id}: paired anchor counts disagree")
        if not all(0 <= finite(summary[key], key) <= 1 for key in ("retrieval_mean", "random_mean")):
            raise ValueError(f"{group_id}: similarity outside [0, 1]")
        gain = by_metric["similarity_gain"]
        for suffix in ("low", "high"):
            same_number(summary.get(f"gain_t_ci95_{suffix}"), gain.get(f"seed_t_ci95_{suffix}"), "gain CI")
        if n == 1 and any(gain.get(f"seed_t_ci95_{s}") is not None for s in ("low", "high")):
            raise ValueError("A one-seed t interval must be empty")
        reports.append(GameReport(game, summary, by_metric, identity, seed_rows))
    return reports


def setting(report, name):
    identity = report.identity
    if name in ("actual_steps", "replay_capacity"):
        value = identity[name]
    elif name == "context_length":
        value = identity["config"]["JointTrainAgent"]["ImagineContextLength"]
    else:
        value = identity["analysis_settings"][name]
    number = int(finite(value, name))
    if number < 0:
        raise ValueError(f"Negative protocol setting {name}")
    return number



def command(name, value):
    if not re.fullmatch(r"Ram[A-Za-z]+", name):
        raise ValueError(f"Invalid numeric macro name: {name}")
    if not re.fullmatch(r"[-0-9.,]*", value):
        raise ValueError(f"Non-numeric macro value: {name}")
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def render_numbers(reports, paper_text):
    """Render numeric definitions and numeric seed-row arguments, never prose."""
    definitions = {}
    active = re.sub(r"(?<!\\)%[^\n]*", "", paper_text)
    used = set(re.findall(r"\\(Ram[A-Za-z]+)\b", active))
    protocol_fields = dict(EvalDecisions="actual_steps", ReplayCapacity="replay_capacity",
                           IndexWarmup="index_warmup", QueryEvery="query_every",
                           ContextLength="context_length", RandomRepeats="random_repeats",
                           RebuildEvery="rebuild_every")
    for report in reports:
        prefix = "Ram" + PREFIXES[report.game]
        s = report.summary
        definitions[prefix + "RetrievalMain"] = f"{s['retrieval_mean']:.3f}"
        definitions[prefix + "RandomMain"] = f"{s['random_mean']:.3f}"
        definitions[prefix + "Rollouts"] = str(int(s['n_seeds']))
        definitions[prefix + "Anchors"] = f"{int(s['paired_queries']):,}"
        for stem, field in (("Retrieval", "retrieval"), ("Random", "random")):
            definitions[prefix + stem + "Mean"] = f"{s[field + '_mean']:.4f}"
            sd = s.get(field + '_seed_sd')
            definitions[prefix + stem + "SD"] = "" if sd is None else f"{sd:.4f}"
            definitions[prefix + stem + "HasSD"] = str(int(sd is not None))
        definitions[prefix + "GainMean"] = f"{s['similarity_gain']:.4f}"
        gain = report.metrics['similarity_gain']
        lo, hi = gain.get('seed_t_ci95_low'), gain.get('seed_t_ci95_high')
        definitions[prefix + "GainCILow"] = "" if lo is None else f"{lo:.4f}"
        definitions[prefix + "GainCIHigh"] = "" if hi is None else f"{hi:.4f}"
        definitions[prefix + "GainHasCI"] = str(int(lo is not None and hi is not None))
        for label, field in protocol_fields.items():
            definitions[prefix + label] = f"{setting(report, field):,}"
        definitions[prefix + "RebuildEnabled"] = str(int(setting(report, 'rebuild_every') > 0))
    for label, field in protocol_fields.items():
        values = [setting(r, field) for r in reports]
        shared = len(set(values)) == 1
        name = "Ram" + label
        if not shared and name in used:
            raise ValueError(f"{field} differs between games. Edit the paper to use per-game numeric macros; wording was left unchanged")
        definitions[name] = f"{values[0]:,}" if shared else ""
    rebuilds = [setting(r, 'rebuild_every') > 0 for r in reports]
    if len(set(rebuilds)) > 1 and 'RamRebuildEnabled' in used:
        raise ValueError("Rebuild settings differ between games; edit the paper's protocol wording before updating")
    definitions['RamRebuildEnabled'] = str(int(all(rebuilds)))
    lines = ["% Generated numeric data. Edit wording and formatting in the manuscript.",
             "% Regenerate with results/update_tex_ram_retrieval.py; no prose is written."]
    lines.extend(command(name, value) for name, value in definitions.items())
    for report in reports:
        prefix = "Ram" + PREFIXES[report.game]
        lines.append("\\newcommand{\\" + prefix + "SeedRows}{%")
        for row in report.seeds:
            # The five arguments are numbers only. The row formatter is authored
            # in the manuscript, including its game label and all punctuation.
            args = [str(int(row['evaluation_seed'])), f"{int(row['paired_queries']):,}",
                    f"{row['retrieval_similarity_mean']:.4f}", f"{row['random_similarity_mean']:.4f}",
                    f"{row['similarity_gain_mean']:.4f}"]
            lines.append("  \\" + prefix + "SeedRow" + "".join("{" + v + "}" for v in args) + "%")
        lines.append("}")
    return "\n".join(lines) + "\n"


def validate_numbers_input(paper_text, tex, numbers):
    if tex.resolve() == numbers.resolve():
        raise ValueError("Numeric output cannot be the manuscript itself")
    if numbers.suffix.lower() != '.tex':
        raise ValueError("Numeric output must be a .tex file")
    inputs = re.findall(r"\\input\s*\{([^}]+)\}", re.sub(r"(?<!\\)%[^\n]*", "", paper_text))
    for name in inputs:
        path = Path(name.strip())
        if not path.suffix:
            path = path.with_suffix('.tex')
        if (tex.parent / path).resolve() == numbers.resolve():
            return
    raise ValueError(f"The manuscript must input the numeric file: {numbers.name}. No manuscript text was changed")

def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".ram_retrieval_", suffix=".tex", delete=False) as stream:
        stream.write(text)
        temporary = Path(stream.name)
    try:
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--excel", type=Path, default=DEFAULT_EXCEL)
    parser.add_argument("--tex", type=Path, default=DEFAULT_TEX, help="Manuscript to read; never edited")
    parser.add_argument("--numbers", type=Path, help="Numeric output; defaults to ram_retrieval_numbers.tex beside the manuscript")
    parser.add_argument("--group", action="append", default=[], metavar="GAME=GROUP_ID")
    parser.add_argument("--refresh-excel", action="store_true")
    parser.add_argument("--inputs", type=Path, nargs="+", help="Input roots forwarded when refreshing Excel")
    parser.add_argument("--dry-run", action="store_true", help="Print the numeric-file diff without writing")
    parser.add_argument("--check", action="store_true", help="Exit 1 if the numeric data needs updating")
    args = parser.parse_args(argv)
    try:
        selections = {}
        for choice in args.group:
            game, sep, group_id = choice.partition("=")
            if not sep or game not in GAMES or not group_id or game in selections:
                raise ValueError("Use each --group at most once, in the form GAME=GROUP_ID")
            selections[game] = group_id
        if args.inputs and not args.refresh_excel:
            raise ValueError("--inputs requires --refresh-excel")
        if args.refresh_excel:
            command = [sys.executable, str(ROOT / "probing/ram_retrieval_eval/export_statistics.py"),
                       "--output", str(args.excel.resolve())]
            if args.inputs:
                command += ["--inputs", *[str(p.resolve()) for p in args.inputs]]
            subprocess.run(command, check=True)
        ram_reader = ROOT / "probing/ram_retrieval_eval/ram_states.py"
        if hashlib.sha256(ram_reader.read_bytes()).hexdigest() != RAM_SHA256:
            raise ValueError("RAM decoder changed; review the appendix's address mapping before updating")
        reports = load_reports(args.excel, selections)
        paper_text = args.tex.read_bytes().decode("utf-8")
        numbers = args.numbers or args.tex.with_name(NUMBERS_FILENAME)
        validate_numbers_input(paper_text, args.tex, numbers)
        before = numbers.read_bytes().decode("utf-8") if numbers.exists() else ""
        after = render_numbers(reports, paper_text)
        if args.dry_run:
            print("".join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                               fromfile=str(numbers), tofile=str(numbers))))
        elif args.check:
            print("Numeric data is current." if before == after else "Numeric data needs an update.")
            return int(before != after)
        elif before != after:
            write_atomic(numbers, after)
            print(f"Updated numeric data: {numbers.resolve()}")
        else:
            print("Numeric data is already current.")
        for report in reports:
            s = report.summary
            print(f"{NAMES[report.game]}: {int(s['n_seeds'])} rollouts, "
                  f"retrieval={s['retrieval_mean']:.4f}, random={s['random_mean']:.4f}")
        return 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    sys.exit(main())
