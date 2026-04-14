#!/usr/bin/env python3
"""
Expand --options-file references in compile_commands.json so clangd can read them.
CMake adds include paths for FetchContent deps via nvcc's --options-file flag, which
clangd doesn't understand. This script inlines those RSP files into the commands.

Run automatically via: cmake --build build --target clangd
"""
import json
import shlex
import sys
from pathlib import Path


def expand_options_files(entry):
    directory = Path(entry["directory"])
    try:
        parts = shlex.split(entry["command"])
    except ValueError:
        return entry

    result = []
    i = 0
    while i < len(parts):
        if parts[i] == "--options-file" and i + 1 < len(parts):
            rsp = directory / parts[i + 1]
            if rsp.exists():
                result.extend(shlex.split(rsp.read_text()))
            i += 2
        else:
            result.append(parts[i])
            i += 1

    entry["command"] = shlex.join(result)
    return entry


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("build/compile_commands.json")
    data = json.loads(path.read_text())
    expanded = [expand_options_files(e) for e in data]
    path.write_text(json.dumps(expanded, indent=2))
    print(f"Expanded {len(data)} entries in {path}")


if __name__ == "__main__":
    main()
