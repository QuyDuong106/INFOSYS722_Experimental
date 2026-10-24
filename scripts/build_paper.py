"""Build the canonical DESRIST paper and enforce its full-paper page range."""
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def main():
    subprocess.run([
        'latexmk', '-pdf', '-outdir=build', '-interaction=nonstopmode',
        '-halt-on-error', 'main.tex'
    ], cwd=ROOT, check=True)
    log = (ROOT/'build/main.log').read_text(errors='replace')
    pages = re.search(r'Output written on .*?\((\d+) pages?', log)
    if not pages or not 13 <= int(pages[1]) <= 18:
        raise RuntimeError('DESRIST full papers require 13–18 pages including references')
    problems = ('Overfull \\hbox', 'Overfull \\vbox', 'undefined', 'multiply defined')
    if any(problem in log for problem in problems):
        raise RuntimeError('Inspect build/main.log for layout or reference problems')
    shutil.move(str(ROOT/'build/main.pdf'), str(ROOT/'main.pdf'))
    print(f'Final PDF: {ROOT / "main.pdf"} ({pages[1]} pages)')

if __name__ == '__main__':
    main()
