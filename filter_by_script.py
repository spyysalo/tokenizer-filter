#!/usr/bin/env python3

# Filter tokenizer vocabulary by script (e.g. only Latin).

import sys
import re
import json
import unicodedata

from collections import defaultdict
from argparse import ArgumentParser

from tokenizers import Tokenizer
from transformers import AutoTokenizer


# Unicode script values to ignore when identifying token script(s)
IGNORED_SCRIPTS = ('Common', 'Unknown')


def argparser():
    ap = ArgumentParser()
    ap.add_argument('tokenizer')
    ap.add_argument('script', nargs='+')
    ap.add_argument('--save-dir', default='filtered-tokenizer')
    ap.add_argument('--verbose', action='store_true')
    return ap


def load_script_data(fn='Scripts.txt'):
    # Parse https://www.unicode.org/Public/UNIDATA/Scripts.txt file,
    # return mapping from code point to script. Adapting in part
    # https://gist.github.com/anonymous/2204527
    SCRIPT_LINE_RE = re.compile(r'^([0-9A-Z]{4,5})(?:\.\.([0-9A-Z]{4,5}))?\s+; ([A-Za-z_]+) #.*')

    script_map = defaultdict(lambda: 'Unknown')
    with open(fn) as f:
        for ln, l in enumerate(f, start=1):
            if l.isspace() or l.startswith('#'):
                continue
            m = SCRIPT_LINE_RE.match(l)
            assert m, f'failed to parse line {ln}: {l}'
            start, end, script = m.groups()
            for i in range(int(start, 16), int(end or start, 16)+1):
                assert i not in script_map, f'dup: {i}'
                script_map[i] = script
    return script_map


def remove_tokens(tokenizer, tokens_to_remove):
    # Adapted in part from https://github.com/1kkiRen/Tokenizer-Changer
    state = json.loads(tokenizer.backend_tokenizer.__getstate__())
    vocab, merges = state['model']['vocab'], state['model']['merges']

    filtered_vocab = {}
    for t, i in sorted(vocab.items(), key=lambda kv: kv[1]):
        if t not in tokens_to_remove:
            filtered_vocab[t] = len(filtered_vocab)
    print(f'Filtered vocab from {len(vocab)} to {len(filtered_vocab)}')

    # Remove merges that create removed tokens
    removed_tokens = set(vocab) - set(filtered_vocab)
    filtered_merges = [m for m in merges if m[0]+m[1] not in removed_tokens]
    print(f'Filtered merges from {len(merges)} to {len(filtered_merges)}')

    # Adjust ids of special tokens (which may or may not be in vocab)
    next_free_id, special_token_id = len(filtered_vocab), {}
    for s in state['added_tokens']:
        if s['content'] in filtered_vocab:
            s['id'] = filtered_vocab[s['content']]
        else:
            s['id'] = next_free_id
            next_free_id += 1
        special_token_id[s['content']] = s['id']
    print(f'Updated ids for {len(special_token_id)} special tokens')

    # Update special token IDs in post-processor

    # Traverse over dict/list structure, calling visit(k,v) on dict elements
    def traverse(obj, visit):
        if isinstance(obj, dict):
            for key, value in obj.items():
                visit(key, value)
                traverse(value, visit)
        elif isinstance(obj, list):
            for item in obj:
                traverse(item, visit)

    # visit function to update ids for special_tokens
    update_count = 0
    def update_special_token_ids(key, value):
        nonlocal update_count
        if key != "special_tokens" or not isinstance(value, dict):
            return
        for k, v in value.items():
            v['ids'] = [special_token_id[t] for t in v['tokens']]
            update_count += 1

    traverse(state['post_processor'], update_special_token_ids)
    print(f'Updated ids for {update_count} special tokens in preprocessor')

    # Build new tokenizer
    state['model']['vocab'] = filtered_vocab
    state['model']['merges'] = filtered_merges
    new_tokenizer = tokenizer.__class__(
        tokenizer_object=Tokenizer.from_str(json.dumps(state)),
        **tokenizer.init_kwargs
    )

    return new_tokenizer


def main(argv):
    args = argparser().parse_args(argv[1:])
    target_scripts = set(args.script)

    script_map = load_script_data()

    known_scripts = set(script_map.values())
    unknown_scripts = target_scripts - known_scripts
    if unknown_scripts:
        print(f'Unknown script(s): {list(unknown_scripts)} (known: {sorted(known_scripts)})')
        return 1

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    # Select tokens to remove
    tokens_to_remove = set()
    for t, i in sorted(tokenizer.vocab.items(), key=lambda kv: kv[1]):
        if i in tokenizer.all_special_ids:
            continue    # keep specials

        token = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        char_script = [script_map[ord(c)] for c in token]
        scripts = set(s for s in char_script if s not in IGNORED_SCRIPTS)

        if scripts - target_scripts:
            if args.verbose:
                print(f'Filtering {i} {token} {scripts}')
            tokens_to_remove.add(t)
    print(f'Removing {len(tokens_to_remove)} tokens')

    # Remove
    new_tokenizer = remove_tokens(tokenizer, tokens_to_remove)

    # Save and sanity-check by testing load
    new_tokenizer.save_pretrained(args.save_dir)
    print(f'Saved tokenizer in {args.save_dir}')
    AutoTokenizer.from_pretrained(args.save_dir)


if __name__ == '__main__':
    sys.exit(main(sys.argv))
