from pathlib import Path
import argparse
import shutil
import zipfile
ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'tests/test_analysis.py', 'tests/test_stotopic_ablation.py'}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--update', action='store_true')
    args = parser.parse_args()
    destination = ROOT / 'dist' / 'stOTopic'
    source_files = [ROOT / name for name in ('.gitignore', 'pyproject.toml', 'MANIFEST.in', 'README.md', 'stOTopic_main.py', 'scripts/export_release.py')]
    for directory in ('stotopic', 'examples', 'tests', '.github', 'tutorials'):
        source_files.extend((p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.relative_to(ROOT).as_posix() not in EXCLUDED and (p.suffix in {'.py', '.yml', '.ipynb', '.md'})))
    if destination.exists() and not args.update:
        raise SystemExit(f'{destination} already exists; choose a new destination or move it first')
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'README.zh-CN.md').unlink(missing_ok=True)
    for relative in EXCLUDED:
        (destination / relative).unlink(missing_ok=True)
    archive = ROOT / 'dist' / 'stOTopic-source.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for source in source_files:
            relative = source.relative_to(ROOT)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            bundle.write(source, Path('stOTopic') / relative)
    print(f'GitHub source directory: {destination}')
    print(f'Source archive: {archive}')
if __name__ == '__main__':
    main()
