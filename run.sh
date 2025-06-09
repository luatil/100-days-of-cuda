#!/bin/bash

# Set build type (default to release if not provided)
buildtype="$2"
if [ -z "$buildtype" ]; then
    buildtype="r"
fi

buildname=""

# Find matching executable if first argument provided
if [ -n "$1" ]; then
    # Use array to handle multiple matches
    matches=(build/*"$1"_*_"$buildtype"*)

    # Check if any matches exist
    if [ -e "${matches[0]}" ]; then
        buildname="${matches[0]}"

        # If multiple matches, use the first one but warn
        if [ ${#matches[@]} -gt 1 ]; then
            echo "Warning: Multiple matches found, using: $buildname"
        fi
    fi
fi

# Handle no matches
if [ -z "$buildname" ]; then
    echo "No matching executable."
    exit 1
fi

# Run the executable with remaining arguments
echo
echo "============ $buildname ============"
echo
"$buildname" "${@:3}"
