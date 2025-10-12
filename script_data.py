#!/usr/bin/env python3

# Parse https://www.unicode.org/Public/UNIDATA/Scripts.txt file,
# return mapping from code point to script. Adapting in part
# https://gist.github.com/anonymous/2204527

import re

from collections import defaultdict


# Regex for parsing Scripts.txt lines up to script definition. Examples:
# 0000..001F    ; Common # Cc  [32] <control-0000>..<control-001F>
# 0020          ; Common # Zs       SPACE
SCRIPT_LINE_RE = re.compile(r'^([0-9A-Z]{4,5})(?:\.\.([0-9A-Z]{4,5}))?\s+; ([A-Za-z_]+) #.*')


def load_script_data(fn='Scripts.txt'):
    # Scripts.txt: "All code points not explicitly listed for Script
    # have the value Unknown"
    script_map = defaultdict(lambda: 'Unknown')
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            if l.isspace() or l.startswith('#'):
                continue    # skip blank and comment lines
            m = SCRIPT_LINE_RE.match(l)
            assert m, f'failed to parse line {ln}: {l}'
            start, end, script = m.groups()
            for i in range(int(start, 16), int(end or start, 16)+1):
                assert i not in script_map, f'dup: {i}'
                script_map[i] = script
    return script_map
