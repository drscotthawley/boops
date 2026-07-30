#!/usr/bin/env python3
"""Rebuild the boops index from every page on gh-pages.

Run by .github/workflows/build-index.yml. This exists because the index is the one page that
depends on *all* the others: boopiter renders each share on the machine that owns it and pushes
only that page, so no machine ever has the full set. The branch does, which makes CI the only
vantage point that can list everything -- and the reason an index built locally kept dropping
whichever pages had been shared from the other machine.
"""
import json, re, shutil, subprocess, sys
from pathlib import Path

import yaml

SITE = Path('site')          # the gh-pages checkout
SRC  = SITE/'.index-src'     # render inputs boopiter publishes alongside its pages
OUT  = Path('build')         # scratch render directory

# Quarto's own preview discovery: it takes the first image in a document and writes it here (the same
# tag behind og:image sharing previews). Read back rather than re-derived, so a plot output counts
# exactly like a pasted photo -- working it out from the notebook instead meant plot-only notebooks
# silently got no thumbnail.
_OG_IMAGE = re.compile(r'<meta[^>]+og:image[^>]+content="([^"]+)"')
# Root files the index owns, written back to the branch. search.json is deliberately absent: this
# render only knows about the index page, so publishing its search index would replace a real one
# covering every page with one covering none.
_WRITE_BACK = ('index.html', 'index.md', 'listings.json', 'sitemap.xml')


def card(name: str) -> dict | None:
    "The listing entry for one published page, or None if it must not be listed."
    meta = SITE/name/'share.json'
    # No sidecar means unlisted -- either boopiter published it that way, or the file was lost. Both
    # resolve to "leave it off", so the failure mode is a missing card rather than an exposed page.
    if not meta.exists(): return None
    try: m = json.loads(meta.read_text())
    except ValueError:
        print(f'  {name}: unreadable share.json, skipping', file=sys.stderr)
        return None
    # Absent `listed` means a sidecar written before the flag existed, and back then a sidecar was only
    # ever written for a public share -- an unlisted one had none at all. So absence defaults to listed:
    # backward compatible without ever promoting a page that was published as unlisted.
    if not m.get('listed', True): return None
    e = {'path': f'{name}/'}
    for k in ('title', 'description', 'date'):
        if m.get(k): e[k] = m[k]
    html = SITE/name/'index.html'
    if html.exists() and (g := _OG_IMAGE.search(html.read_text(errors='replace'))):
        e['image'] = f'{name}/{g.group(1)}'   # og:image is relative to its own page, the listing to the root
    return e


def main() -> None:
    if not SRC.is_dir():
        sys.exit(f'{SRC} not found -- share once from boopiter so it publishes the index render inputs')
    pages = [p.name for p in sorted(SITE.iterdir())
             if p.is_dir() and p.name != 'site_libs' and not p.name.startswith(('.', '_'))]
    items = [c for c in (card(n) for n in pages) if c]
    print(f'{len(items)} of {len(pages)} page(s) listed: {[i["path"] for i in items]}')

    shutil.rmtree(OUT, ignore_errors=True)
    shutil.copytree(SRC, OUT)
    idx = (OUT/'index.qmd').read_text()
    # boopiter's local index globs for the shares it can render; here every page is given inline
    # instead, with its own metadata. Quarto accepts a listing item either way, which is what lets a
    # card point at a page this render has no source for -- all of them, in fact.
    glob = '  contents: "*/index.ipynb"'
    if glob not in idx: sys.exit(f'{OUT}/index.qmd has no {glob!r} line to replace')
    body = yaml.safe_dump(items, sort_keys=False, allow_unicode=True).rstrip() if items else '[]'
    contents = '  contents:\n' + '\n'.join('    ' + l for l in body.splitlines()) if items \
               else '  contents: []'
    (OUT/'index.qmd').write_text(idx.replace(glob, contents))

    subprocess.run(['quarto', 'render'], cwd=OUT, check=True)

    built = OUT/'_site'
    for f in _WRITE_BACK:
        if (built/f).exists(): shutil.copy(built/f, SITE/f)
    # site_libs is shared by the index and every page; merged rather than replaced so a page's assets
    # survive even if this render emits a different subset.
    if (built/'site_libs').is_dir():
        shutil.copytree(built/'site_libs', SITE/'site_libs', dirs_exist_ok=True)
    print('index rebuilt')


if __name__ == '__main__':
    main()
