#!/usr/bin/env python3
"""Restore pinned public source snapshots and the paper. No weights or training."""
import hashlib
import json
import posixpath
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def download(url, dest, expected):
    request = urllib.request.Request(url, headers={'User-Agent': 'MiSS-project-setup/1.0'})
    with urllib.request.urlopen(request, timeout=120) as response, dest.open('wb') as out:
        shutil.copyfileobj(response, out)
    if digest(dest) != expected:
        raise RuntimeError('Downloaded SHA256 mismatch: ' + url)


def extract_regular_files(archive, destination):
    # Materialize internal file links; never create filesystem symlinks.
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        roots = {PurePosixPath(m.name).parts[0] for m in members if m.name}
        if len(roots) != 1:
            raise RuntimeError('Expected a single archive root')
        archive_root = next(iter(roots))
        for member in members:
            parts = PurePosixPath(member.name).parts
            if member.name.startswith('/') or '..' in parts:
                raise RuntimeError('Unsafe archive path')
            if len(parts) < 2:
                continue
            target = destination.joinpath(*parts[1:])
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile() or member.issym() or member.islnk():
                source_member = member
                visited = set()
                while source_member.issym() or source_member.islnk():
                    if source_member.name in visited:
                        raise RuntimeError('Cyclic archive link')
                    visited.add(source_member.name)
                    link_name = source_member.linkname
                    if source_member.issym():
                        link_name = posixpath.join(posixpath.dirname(source_member.name), link_name)
                    link_name = posixpath.normpath(link_name)
                    if not link_name.startswith(archive_root + '/'):
                        raise RuntimeError('Archive link escapes source root')
                    source_member = tar.getmember(link_name)
                if not source_member.isfile():
                    raise RuntimeError('Archive link must resolve to a regular file')
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(source_member) as source, target.open('wb') as out:
                    shutil.copyfileobj(source, out)
                target.chmod(source_member.mode & 0o777)
            else:
                raise RuntimeError('Links or special files are not accepted: ' + member.name)


def check_snapshot(folder, prefix, checksums):
    for relative, expected in checksums.items():
        if relative.startswith(prefix + '/'):
            path = folder / relative[len(prefix) + 1:]
            if not path.is_file() or digest(path) != expected:
                raise RuntimeError('Source file missing or changed: ' + relative)


def main():
    lock = json.loads((ROOT / 'sources.lock.json').read_text())
    checksums = json.loads((ROOT / 'third_party/SHA256SUMS.json').read_text())['files']
    cache = ROOT / '.cache'
    cache.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='sources-', dir=cache) as work:
        work = Path(work)
        for repo in lock['repositories']:
            target = ROOT / repo['path']
            if target.exists():
                check_snapshot(target, repo['path'], checksums)
                print('Verified existing:', repo['path'])
                continue
            archive = work / (repo['name'] + '.tar.gz')
            stage = work / repo['name']
            stage.mkdir()
            download(repo['archive_url'], archive, repo['archive_sha256'])
            extract_regular_files(archive, stage)
            check_snapshot(stage, repo['path'], checksums)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(stage), str(target))
            print('Restored:', repo['path'])
        paper = lock['paper']
        dest = ROOT / paper['path']
        expected = lock['downloaded_files_sha256'][paper['path']]
        if dest.exists():
            if digest(dest) != expected:
                raise RuntimeError('Existing paper has changed; will not overwrite')
        else:
            temp = work / 'paper.pdf'
            download(paper['url'], temp, expected)
            if not temp.read_bytes().startswith(b'%PDF-'):
                raise RuntimeError('Downloaded paper is not a PDF')
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp), str(dest))
        print('Verified paper:', paper['path'])
    print('Run python3 scripts/validate_workspace.py next. No training was started.')


if __name__ == '__main__':
    main()
