#!/usr/bin/env python3
"""Run the unchanged I006 case through NumPilot's real AST/NPU mainline.

Load CANN, then use the same interpreter as the official qualification:
  python tools/qualified_i006_migration.py --capability-overlay /abs/overlay.json \
      --evidence-root /abs/evidence --runtime-site /abs/runtime/site --out /abs/new-run

This is one sample's scoped migration evidence, not a whole-runtime audit,
150-case benchmark promotion, or performance measurement. No LLM is called.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RELATIVE = 'benchmark/cases/I_gap_model/I006_concatenate_axis/program.py'
SOURCE_SHA256 = '17909c8f44bfc183a61e068c96e5df5d5f5e3678e8a8e854f3cc67246fd3823b'
EXPECTED_REWRITES = {'concatenate': 1}


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write('\n')


def _inside(path, parent: Path) -> bool:
    try:
        Path(path).resolve(strict=True).relative_to(parent)
        return True
    except (OSError, ValueError, TypeError):
        return False


def select_runtime_site(site: Path) -> str:
    """Select the qualification wheel before any AsNumpy import can occur."""
    site = site.resolve(strict=True)
    if not site.is_dir():
        raise ValueError('runtime-site must be a directory')
    for name, module in tuple(sys.modules.items()):
        if name == 'asnumpy' or name.startswith('asnumpy.'):
            if not _inside(getattr(module, '__file__', None), site):
                raise ValueError('AsNumpy was already imported outside runtime-site')
    sys.path.insert(0, str(site))
    spec = importlib.util.find_spec('asnumpy')
    if spec is None or not _inside(spec.origin, site):
        raise ValueError('AsNumpy import does not resolve inside runtime-site')
    return str(Path(spec.origin).resolve())


def assess_evidence(result: dict, *, profiles: dict, overlay: dict,
                    runtime_before: dict, runtime_after: dict,
                    source_sha256: str, analysis_sha256: str,
                    source_unchanged: bool) -> list[str]:
    """Require all mainline facts before issuing this sample's scoped receipt."""
    problems = []

    def require(condition, reason):
        if not condition:
            problems.append(reason)

    require(result.get('status') == 'pass', 'mainline verification did not pass')
    require(result.get('engine') == 'ast', 'migration did not use the AST engine')
    require(result.get('rewrite_integrity') is True, 'rewrite integrity failed')
    require(result.get('rewrites') == EXPECTED_REWRITES, 'unexpected rewrite set')
    require(result.get('expected_rewrites') == EXPECTED_REWRITES,
            'analysis did not require exactly the I006 concatenate call')
    require(set(EXPECTED_REWRITES) <= set(overlay.get('operators', [])),
            'actual operator is outside the official overlay')
    require(source_unchanged and source_sha256 == SOURCE_SHA256,
            'original benchmark source changed')
    require(result.get('source_sha256') == source_sha256,
            'mainline oracle is not the original benchmark source')
    require(result.get('analysis_source_sha256') == analysis_sha256,
            'analysis source binding mismatch')
    generated = result.get('migrated_source')
    require(isinstance(generated, str) and bool(generated), 'generated source missing')
    require(runtime_before == runtime_after, 'runtime identity changed during migration')
    require(runtime_before.get('identity_sha256') == overlay.get('runtime_identity_sha256')
            and bool(runtime_before.get('identity_sha256')),
            'runtime identity does not match official overlay')
    require(profiles.get('matched_callsites') == profiles.get('attempted_callsites') == 1,
            'exact I006 callsite was not profiled')
    profile_set = profiles.get('profiles') or {}
    require(profile_set.get('source_sha256') == analysis_sha256,
            'profile source binding mismatch')
    sites = profile_set.get('callsites') or []
    require(len(sites) == 1 and sites[0].get('qualified_name') == 'numpy.concatenate',
            'unexpected profiled callsite')
    expected_profile = sites[0].get('profile') if len(sites) == 1 else None
    evidence = result.get('validation_evidence') or {}
    require(evidence.get('status') == 'pass' and evidence.get('backend') == 'npu',
            'verification is not a passing NPU execution')
    for fact in ('numerically_consistent', 'input_consistent', 'kernel_consistent',
                 'artifact_consistent', 'artifact_inputs_executed'):
        require(evidence.get(fact) is True, 'verification fact failed: ' + fact)
    require(evidence.get('source_sha256') == source_sha256, 'oracle evidence hash mismatch')
    require(isinstance(generated, str) and
            evidence.get('migrated_sha256') == _sha(generated.encode('utf-8')),
            'generated artifact evidence hash mismatch')
    comparisons = evidence.get('comparisons') or {}
    for name in ('inputs', 'kernel', 'artifact'):
        require((comparisons.get(name) or {}).get('consistent') is True,
                name + ' comparison failed or missing')
    guard = evidence.get('backend_provenance') or {}
    require(guard.get('backend') == 'npu', 'guard backend is not NPU')
    for fact in ('strict_npu', 'ok', 'preflight_ok', 'preflight_strict_npu',
                 'runtime_profile_ok', 'runtime_target_integrity_ok',
                 'runtime_execution_ok', 'runtime_trace_ok'):
        require(guard.get(fact) is True, 'strict runtime fact failed: ' + fact)
    require(guard.get('profile_status') == 'matched', 'runtime profile is not matched')
    for field in ('required_ops', 'profile_scoped'):
        require(guard.get(field) == ['concatenate'], 'unexpected guarded scope: ' + field)
    for field in ('missing', 'non_strict', 'runtime_target_integrity_errors'):
        require(guard.get(field) == [], 'guard reported issues: ' + field)
    counts = guard.get('runtime_call_counts') or {}
    for path in ('kernel', 'artifact'):
        require(counts.get(path) == EXPECTED_REWRITES,
                path + ' did not dispatch exactly the original callsite')
    checks = guard.get('runtime_profile_checks') or {}
    calls = checks.get('concatenate') or []
    require(set(checks) == {'concatenate'} and len(calls) == 2,
            'guard did not record both actual concatenate calls')
    for index, check in enumerate(calls):
        require(check.get('matched') is True and check.get('profile') == expected_profile,
                'actual call profile diverged from original I006: ' + str(index))
    return problems


def run(args) -> dict:
    out = Path(args.out).expanduser().absolute()
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    receipt = {
        'schema_version': 1,
        'kind': 'numpilot_scoped_sample_migration',
        'sample_id': 'I006_concatenate_axis',
        'status': 'blocked',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'identity_scope': 'official_overlay_callsite',
        'global_runtime_audit_promoted': False,
        'benchmark_suite_updated': False,
        'performance_valid': False,
        'model_called': False,
        'execution': 'Orchestrator AST migration + strict NPU + original NumPy oracle',
        'argv': list(sys.argv),
        'artifacts': {},
    }
    try:
        # Keep the repository importable when launched as a standalone script.
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        site = Path(args.runtime_site).expanduser().resolve(strict=True)
        origin = select_runtime_site(site)
        from numpilot.benchmark_lowering import lower_benchmark_program
        from numpilot.benchmark_overlay import load_benchmark_overlay, build_analysis_profiles
        from numpilot.engines.ast_rewrite import AstRewriteEngine
        from numpilot.orchestrator import Orchestrator

        source_path = REPO_ROOT / SOURCE_RELATIVE
        source_bytes = source_path.read_bytes()
        source_hash = _sha(source_bytes)
        if source_hash != SOURCE_SHA256:
            raise ValueError('I006 source hash changed; review the benchmark before running')
        source = source_bytes.decode('utf-8')
        overlay_path = Path(args.capability_overlay).expanduser().resolve(strict=True)
        evidence_root = Path(args.evidence_root).expanduser().resolve(strict=True)
        overlay = load_benchmark_overlay(str(overlay_path), str(evidence_root))
        if 'concatenate' not in overlay.operators:
            raise ValueError('official overlay does not admit concatenate')
        runtime_before = overlay.resolver.runtime_identity().to_dict()
        if not _inside(runtime_before.get('module_path'), site):
            raise ValueError('runtime identity is not bound to runtime-site')
        if runtime_before['identity_sha256'] != overlay.runtime_identity_sha256:
            raise ValueError('runtime identity changed after overlay admission')
        lowered = lower_benchmark_program(source)
        analysis_hash = _sha(lowered.source.encode('utf-8'))
        collected = build_analysis_profiles(lowered.source, overlay)
        profiles = {
            'matched_callsites': collected.matched_callsites,
            'attempted_callsites': collected.attempted_callsites,
            'notes': list(collected.notes),
            'profiles': collected.profiles.to_dict() if collected.profiles else None,
        }
        # BenchmarkOverlay's default description describes its usual shim consumer.
        # This entry uses its real resolver; preserve that execution distinction.
        admission = {**overlay.to_dict(), 'execution': 'strict_npu_mainline'}
        (out / 'original.py').write_bytes(source_bytes)
        (out / 'analysis_source.py').write_text(lowered.source, encoding='utf-8')
        _write_json(out / 'lowering.json', lowered.to_dict())
        _write_json(out / 'profiles.json', profiles)
        _write_json(out / 'overlay-admission.json', admission)
        _write_json(out / 'runtime-before.json', runtime_before)
        receipt.update({
            'original_source': str(source_path), 'source_sha256': source_hash,
            'analysis_source_sha256': analysis_hash, 'runtime_site': str(site),
            'runtime_import_origin': origin, 'capability_overlay': str(overlay_path),
            'evidence_root': str(evidence_root),
            'qualification_id': overlay.qualification_id,
            'overlay_sha256': overlay.overlay_sha256,
            'runtime_identity_sha256': runtime_before['identity_sha256'],
            'global_runtime_status': runtime_before['status'],
            'global_runtime_issues': runtime_before['issues'],
        })
        if collected.profiles is None or collected.matched_callsites != 1 or collected.attempted_callsites != 1:
            raise ValueError('exact original I006 profile is unavailable')
        result = Orchestrator(
            backend=overlay.resolver, engine=AstRewriteEngine(),
            analysis_profiles=collected.profiles,
        ).run_source(
            lowered.source, 'program.py', out_dir=str(out / 'mainline'),
            verification_source=source, strict_input_types=True,
            prepare_source=False,
        )
        runtime_after = overlay.resolver.runtime_identity().to_dict()
        _write_json(out / 'runtime-after.json', runtime_after)
        evidence = result.get('validation_evidence') or {}
        _write_json(out / 'validation-evidence.json', evidence)
        _write_json(out / 'runtime-guard.json', evidence.get('backend_provenance') or {})
        _write_json(out / 'comparisons.json', evidence.get('comparisons') or {})
        problems = assess_evidence(
            result, profiles=profiles, overlay=admission,
            runtime_before=runtime_before, runtime_after=runtime_after,
            source_sha256=source_hash, analysis_sha256=analysis_hash,
            source_unchanged=source_path.read_bytes() == source_bytes,
        )
        receipt.update({
            'status': 'passed' if not problems else 'blocked',
            'scoped_qualified': not problems,
            'mainline_status': result.get('status'), 'problems': problems,
            'expected_rewrites': result.get('expected_rewrites'),
            'rewrites': result.get('rewrites'),
            'generated_source_sha256': evidence.get('migrated_sha256'),
            'claim': 'I006 sample automatically migrated and independently verified on NPU'
                     if not problems else 'I006 strict migration acceptance was not established',
        })
    except Exception as exc:
        receipt.update({'scoped_qualified': False,
                        'problems': [type(exc).__name__ + ': ' + str(exc)]})
    receipt['artifacts'] = {
        str(path.relative_to(out)): _sha(path.read_bytes())
        for path in sorted(out.rglob('*')) if path.is_file()
    }
    _write_json(out / 'receipt.json', receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capability-overlay', required=True)
    parser.add_argument('--evidence-root', required=True)
    parser.add_argument('--runtime-site', required=True)
    parser.add_argument('--out', required=True, help='new output directory; never overwrites evidence')
    args = parser.parse_args(argv)
    try:
        receipt = run(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, str(exc) + '\n')
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0 if receipt['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
