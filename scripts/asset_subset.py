#!/usr/bin/env python3
"""Enumerate the RoboCasa asset subset the RoboQuest tasks need, then list, link or pack it.

The six official packs total about 24 GB. The kitchens and tasks use four of them, and only the object
categories the registries and the task code name: fixtures and textures (every kitchen), every lightwheel
object (kitchen layouts bake accessories in), and the objaverse / aigen_objs categories found by scanning
suite/v1 and roboquest. Generative textures are never used (no kitchen is built with
generative_textures set). Result: about 7.5 GB (the same for development and
evaluation instances, whose held-out styles add 436 MB of generative textures).

    scripts/asset_subset.py                                    # table of entries and sizes
    scripts/asset_subset.py --list subset.txt                  # relative paths, one per line (tar -T / rsync --files-from)
    scripts/asset_subset.py --link-tree /data/robocasa-assets  # symlink tree usable as bootstrap --assets link:DIR
    scripts/asset_subset.py --tar - | ssh host 'mkdir -p ~/robocasa-assets && tar -C ~/robocasa-assets -xf -'

Plain python3, no simulator imports. --assets-root defaults to the RoboCasa source tree of
$ROBOQUEST_RUNTIME (or the shared runtime path when that is unset).
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME = os.environ.get('ROBOQUEST_RUNTIME', os.path.expanduser('~/robot-agent-runtime'))
BASE_ENTRIES = ('fixtures', 'textures', 'objects/lightwheel')
CATEGORY = re.compile(r'(objaverse|aigen_objs)/([A-Za-z_]+)')
BARE_MODEL = re.compile(r'\b([a-z][a-z_]*)/\1_\d+\b')       # 'tangerine/tangerine_3': a model named without its pack
PACKS = ('objaverse', 'aigen_objs')


def referenced_categories(registry_root, code_roots, assets_root):
    found = set()
    scans = [(registry_root, ('*.json',))] + [(root, ('*.py',)) for root in code_roots]
    for root, patterns in scans:
        for pattern in patterns:
            for path in Path(root).rglob(pattern):
                try:
                    text = path.read_text(errors='ignore')
                except OSError:
                    continue
                for pack, category in CATEGORY.findall(text):
                    found.add(f'objects/{pack}/{category}')
                for category in set(BARE_MODEL.findall(text)):
                    for pack in PACKS:
                        if (assets_root / 'objects' / pack / category).is_dir():
                            found.add(f'objects/{pack}/{category}')
    return sorted(found)


def style_generative_textures(assets_root):
    """Generative-texture files the kitchen styles reference. Styles 11 and up (the held-out evaluation styles)
    name textures like 'gentex042' or 'top_gentex041'; the fixture registries map each name to
    generative_textures/<kind>/texNNN.png. Without them every evaluation scene fails at reset."""
    mapping = {}
    for registry in (assets_root / 'fixtures/fixture_registry').glob('*.yaml'):
        name = None
        for line in registry.read_text(errors='ignore').splitlines():
            head = re.match(r'^([A-Za-z0-9_]+):\s*$', line)
            if head:
                name = head.group(1)
            texture = re.search(r'(generative_textures/[^\s"\']+)', line)
            if texture and name:
                mapping.setdefault(name, set()).add(texture.group(1))
    files = set()
    for style in (assets_root / 'scenes/kitchen_styles').glob('*/style*.yaml'):
        for name in set(re.findall(r'\b([A-Za-z_]*gentex\d+)\b', style.read_text(errors='ignore'))):
            files |= mapping.get(name, set())
    return sorted(files)


def tree_size(path):
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--assets-root', type=Path, default=Path(DEFAULT_RUNTIME) / 'src/robocasa/robocasa/models/assets')
    parser.add_argument('--registry-root', type=Path, default=REPO / 'suite/v1')
    parser.add_argument('--code-root', type=Path, action='append', default=None,
                        help='python trees to scan for object categories (default: roboquest and scripts)')
    parser.add_argument('--extra', action='append', default=[], help='extra entry relative to the assets root')
    parser.add_argument('--list', type=Path, help='write the entries, one relative path per line')
    parser.add_argument('--link-tree', type=Path, help='build a directory of symlinks to the entries')
    parser.add_argument('--tar', help='write an uncompressed tar of the entries to this file, or - for stdout')
    args = parser.parse_args()

    code_roots = args.code_root or [REPO / 'roboquest', REPO / 'scripts']
    entries = list(BASE_ENTRIES) + referenced_categories(args.registry_root, code_roots, args.assets_root) + list(args.extra)
    entries = list(dict.fromkeys(entries + style_generative_textures(args.assets_root)))
    present, missing, total = [], [], 0
    for entry in entries:
        path = args.assets_root / entry
        if path.is_dir():
            present.append(entry)
            total += tree_size(path)
        elif path.is_file():
            present.append(entry)
            total += path.stat().st_size
        else:
            missing.append(entry)
    if args.tar != '-':
        print(f'assets root {args.assets_root}')
        textures = [e for e in present if e.startswith('generative_textures/')]
        for entry in present:
            if entry not in textures:
                print(f'{tree_size(args.assets_root / entry) / 1e6:9.0f} MB  {entry}')
        if textures:
            size = sum((args.assets_root / e).stat().st_size for e in textures)
            print(f'{size / 1e6:9.0f} MB  generative_textures: {len(textures)} files the kitchen styles reference')
        print(f'{total / 1e9:.1f} GB in {len(present)} entries')
        for entry in missing:
            print(f'MISSING under the assets root: {entry}', file=sys.stderr)

    if args.list:
        args.list.write_text('\n'.join(present) + '\n')
        if args.tar != '-':
            print(f'wrote {args.list}')
    if args.link_tree:
        for entry in present:
            target = args.link_tree / entry
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to((args.assets_root / entry).resolve())
        if args.tar != '-':
            print(f'linked {len(present)} entries under {args.link_tree}')
    if args.tar:
        command = ['tar', '-C', str(args.assets_root), '-hcf', args.tar] + present
        sys.exit(subprocess.call(command))
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
