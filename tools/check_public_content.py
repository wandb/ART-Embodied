"""Reject private research paths and marked internal content before publication."""

import argparse
from pathlib import PurePosixPath
import subprocess
import sys

PRIVATE_PARTS = {".research", "scratch", ".private", "internal-notes", "private-notes"}
MARKER = b"ART-EMBODIED " + b"INTERNAL ONLY"


def git(*args):
    return subprocess.check_output(["git", *args])


def private_path(path):
    parts = PurePosixPath(path).parts
    return bool(set(parts) & PRIVATE_PARTS) or any(
        parts[i : i + 2] in (("docs", "research"), ("docs", "proposals"))
        for i in range(len(parts) - 1)
    )


def inspect(paths, objects):
    errors = [f"Private path: {p}" for p in sorted(set(paths)) if private_path(p)]
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE
    )
    try:
        for oid in sorted(set(objects)):
            process.stdin.write(oid + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3:
                raise RuntimeError("Cannot inspect Git object")
            size = int(header[2])
            # Stream large objects and retain overlap for markers crossing chunks.
            tail, found = b"", False
            while size:
                chunk = process.stdout.read(min(size, 65536))
                if not chunk:
                    raise RuntimeError("Truncated Git object")
                found |= MARKER in tail + chunk
                tail = (tail + chunk)[-len(MARKER) :]
                size -= len(chunk)
            if process.stdout.read(1) != b"\n":
                raise RuntimeError("Invalid Git object framing")
            if header[1] == b"blob" and found:
                errors.append(f"Internal-only marker in blob {oid.decode()}")
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    if process.returncode:
        raise RuntimeError("Git object scan failed")
    return errors


def staged():
    paths, objects = [], []
    for entry in git("ls-files", "--stage", "-z").split(b"\0"):
        if not entry:
            continue
        metadata, path = entry.split(b"\t", 1)
        mode, oid, _stage = metadata.split()
        paths.append(path.decode("utf-8", errors="surrogateescape"))
        if mode != b"160000":
            objects.append(oid)
    return inspect(paths, objects)


def history(revisions):
    paths = [
        p.decode("utf-8", errors="surrogateescape").strip("\n")
        for p in git("log", "--format=", "--name-only", "-z", *revisions).split(b"\0")
        if p
    ]
    objects = [
        row.split(b" ", 1)[0]
        for row in git("rev-list", "--objects", *revisions).splitlines()
    ]
    return inspect(paths, objects)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--staged", action="store_true")
    modes.add_argument("--history", metavar="REVISION")
    modes.add_argument("--pre-push", action="store_true")
    args = parser.parse_args()
    errors = []
    if args.staged:
        errors = staged()
    elif args.history:
        errors = history([args.history])
    else:
        for line in sys.stdin:
            _local_ref, local_sha, _remote_ref, remote_sha = line.split()
            if set(local_sha) == {"0"}:
                continue
            revisions = [local_sha]
            if (
                set(remote_sha) != {"0"}
                and subprocess.run(
                    ["git", "cat-file", "-e", remote_sha],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                ).returncode
                == 0
            ):
                revisions += ["--not", remote_sha]
            errors += history(revisions)
    if errors:
        print(
            "Publication blocked:\n" + "\n".join(sorted(set(errors))), file=sys.stderr
        )
        return 1
    print("Public-content check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
