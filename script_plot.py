#!/usr/bin/env python

# Plot token Unicode script distribution.

import sys
import math

import matplotlib.pyplot as plt

from collections import Counter
from argparse import ArgumentParser

from transformers import AutoTokenizer

from script_data import load_script_data


SCRIPT_MAP = load_script_data()


class NonUnicodeCategory:
    special = '[special]'
    mixed = '[mixed]'
    other = '[other]'


COLOR_MAP = {
    'Arabic': '#2ca02c',
    'Armenian': '#ed9809',
    'Bengali': '#17becf',
    'Cyrillic': '#ffd10a',
    'Devanagari': '#ff7f0e',
    'Georgian': '#fa0006',
    'Greek': '#1f77b4',
    'Gurmukhi': '#ff9896',
    'Han': '#d62728',
    'Hebrew': '#0d62ab',
    'Latin': '#010988',
    'Tamil': '#20ff88',
    'Telugu': '#98df8a',
    'Thai': '#940022',
    'Common': '#eeeeee',
    NonUnicodeCategory.special: 'purple',
    NonUnicodeCategory.mixed: '#e377c2',
    NonUnicodeCategory.other: '#cccccc',
}


def argparser():
    ap = ArgumentParser()
    ap.add_argument('tokenizer')
    ap.add_argument('--min-count', type=int, default=100,
                    help='min occurrences not to fold into "other"')
    ap.add_argument('--title')
    ap.add_argument('--plotfile', help='filename')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--exclude', nargs='+', default=None)
    return ap


def categorize(token):
    categories = set(SCRIPT_MAP[ord(c)] for c in token)
    if len(categories) == 1:
        return list(categories)[0]
    else:
        return NonUnicodeCategory.mixed


def combine_rare_counts(counts, min_count, other=NonUnicodeCategory.other):
    separate = {
        k: v for k, v in counts.items()
        if v >= min_count and k != other
    }
    other_total = sum(
        v for k, v in counts.items()
        if v < min_count or k == other
    )
    return Counter(separate) + Counter({ other: other_total })


def main(argv):
    args = argparser().parse_args(argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    counts = Counter()
    for i in sorted(tokenizer.vocab.values()):
        if i in tokenizer.all_special_ids:
            counts[NonUnicodeCategory.special] += 1
        else:
            token = tokenizer.decode([i], clean_up_tokenization_spaces=False)
            if len(token) > 1 and token[0] == ' ' and not token[1].isspace():
                token = token[1:]    # drop token-initial space
            category = categorize(token)
            counts[category] += 1
            if args.verbose:
                print(token, category, file=sys.stderr)

    if args.exclude is not None:
        for l in args.exclude:
            if l in counts:
                del counts[l]

    def print_counts(m):
        print(m)
        for l in [k for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]:
            print(l, counts[l])

    print_counts('before combining rare')
    counts = combine_rare_counts(counts, args.min_count)
    print_counts('-'*78 + '\nafter combining rare')

    labels = [k for k, v in sorted(counts.items(), key=lambda kv: -kv[1])]

    # plot
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(
        [counts[l] for l in labels],
        colors=[COLOR_MAP.get(l, 'black') for l in labels],
        startangle=140,
        wedgeprops=dict(width=0.5, edgecolor='w')
    )

    total = sum(counts.values())
    label_text = { k: f"{k} ({v}, {v/total:.1%})" for k, v in counts.items() }
    for i, (w, l) in enumerate(zip(wedges, labels)):
        angle = (w.theta1+w.theta2)/2
        x = math.cos(math.radians(angle))
        y = math.sin(math.radians(angle))
        style = "angle,angleA=0,angleB={}".format(angle)
        ax.annotate(
            label_text[l],
            xy=(x, y),
            xytext=(x*1.2, y*1.2),
            ha='left' if x > 0 else 'right',
            va='center',
            fontsize=10,
            arrowprops=dict(arrowstyle="-", connectionstyle=style)
        )

    #ax.legend(wedges, labels, loc="center left", bbox_to_anchor=(1, 0.5))
    if args.title is not None:
        title = args.title
    else:
        title = args.tokenizer
        if args.exclude is not None:
            title += ' (excluding ' + ','.join(args.exclude) + ')'
    plt.title(title, fontsize=14)
    plt.tight_layout()
    if args.plotfile is not None:
        plt.savefig(args.plotfile)
    else:
        plt.show()


if __name__ == '__main__':
    sys.exit(main(sys.argv))
