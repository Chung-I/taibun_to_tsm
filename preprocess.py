import argparse
import re
import gzip
from pathlib import Path
from collections import Counter
from functools import partial
import logging
import json
import zhon.hanzi

arabic_chinese_mapping = {
    0: "零",
    1: "一",
    2: "二",
    3: "三",
    4: "四",
    5: "五",
    6: "六",
    7: "七",
    8: "八",
    9: "九",
}
diacritical = 'A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u00ff\u0100-\u017f\u0180-\u024f\u0300-\u036f'

def strip_puncts(a):
    a=a.replace('-',' ')
    a=a.replace("，"," ")
    a=a.replace(","," ")
    a=a.replace(".", " ")
    a=a.replace("?", " ")
    a=a.replace("﹖", " ")
    a=a.replace("!", " ")
    a=a.replace(":", " ")
    a=a.replace("”", " ")
    a=a.replace('“', ' ')
    a=a.replace(";", " ")
    a=a.replace("/", " ")
    a=a.replace("…", " ")
    a=a.replace("(", " ")
    a=a.replace(")", " ")
    a=a.replace("‘", " ")
    a=a.replace("’", " ")
    a=a.replace("／", " ")
    a=a.replace("⋯", " ")
    a=a.replace("~", " ")
    a=a.replace("（", " ")
    a=a.replace("）", " ")
    a=a.replace("「", " ")
    a=a.replace("」", " ")
    a=a.replace("％", " ")
    a=a.replace("｣", " ")
    a=a.replace("%", " ")
    a=a.replace("、", " ")
    a=a.replace("｢", " ")
    a=a.replace("--", " ")
    a=a.replace("！", " ")
    a=a.replace("：", " ")
    a=a.replace("。", " ")
    a=a.replace("—", " ")
    a=a.replace("？", " ")
    a=a.replace("；", " ")
    a=a.replace("──", " ")
    a=a.replace("\"", " ")
    a=a.replace("\'", " ")
    a=a.replace("^", " ")
    a=a.replace("  ", " ")
    a=a.strip()
    return a

def flatten(l):
    return [item for sublist in l for item in sublist]

def convert_arabic_number_to_chinese(sent):
    sent = "".join([arabic_chinese_mapping[int(char)] if re.match("\d", char) else char for char in sent])
    return sent

def read_gzip_file(gzip_file):
    with gzip.open(gzip_file, 'rb') as fp:
        raw_text = fp.read().decode('utf-8')
    return raw_text

def parse_taibun_sent(sent):
    sent = sent.lower()
    sent = re.sub("[^\u4e00-\u9fa5A-Za-z0-9]", " ", sent) 
    words = sent.split()
    return words

def parse_tsm_word(word, is_taibun=False, split=True):
    if not split:
        return [word]
    syls = word.split("-")
    def parse_tgt_syl(syl):
        if re.match("^\d+$", syl):
            return " ".join(syl)
        else:
            syl = re.sub("[^\u4e00-\u9fa5A-Za-z0-9]", "", syl)
            try:
                return re.match("\d?(\w+\d)", syl).group(1)
            except AttributeError:
                if is_taibun:
                    return syl
                return ""
    return list(filter(lambda word: re.sub("\s+", "", word), map(parse_tgt_syl, syls)))

def parse_sent(sent, split=True):
    sent = sent.lower()
    pairs = []
    for word in re.split("\s+", sent):
        taibun, tsm = word.split("｜")
        parsed_taibun = parse_tsm_word(taibun, is_taibun=True, split=split)
        parsed_tsm = parse_tsm_word(tsm, split=split)
        if len(parsed_taibun) != len(parsed_tsm):
            print(parsed_taibun, parsed_tsm)
            return ""
        pairs += list(zip(parsed_taibun, parsed_tsm))
    sent = " ".join([f"{x[0]}|{x[1]}" for x in pairs])
    return sent

def write_lines_to_file(lines, text_file):
    with open(text_file, 'w') as fp:
        for line in lines:
            fp.write(line + '\n')

def get_all_datas(input_dir, prefixes):
    sents = []
    for prefix in prefixes:
        text = read_gzip_file(Path(input_dir).joinpath(f"{prefix}.txt.gz"))
        sents += list(filter(lambda line: line, text.splitlines()))
    return sents

def process_tat_sent(jsonfile):
    with open(jsonfile) as fp:
        trn = json.load(fp)

    taibun = trn['漢羅台文']
    taibun = re.sub("\(.*?\)", " ", taibun)
    taibun = re.sub(f"\/[{diacritical}]+\d?", " ", taibun)

    tailo = trn['台羅數字調']
    tailo = re.sub(f"\/.*?\s+?", " ", tailo)

    taibun_words = [match.group(0) for match in re.finditer(f"([{zhon.hanzi.characters}]|[{diacritical}]+|\d+|%)", taibun)]
    tailo_words = [match.group(0) for match in re.finditer("(\w+\d)", tailo)]
    if len(taibun_words) != len(tailo_words):
        print(jsonfile, taibun, taibun_words, tailo_words)
        return []
    return " ".join(["|".join(pair) for pair in zip(taibun_words, tailo_words)])

def get_tat_datas(input_dir):
    jsons = {}
    for jsonfile in Path(input_dir).rglob("*.json"):
        jsons[jsonfile.name] = jsonfile
    list_word_pairs = [process_tat_sent(jsonfile) for jsonfile in jsons.values()]
    return list_word_pairs

def main(args):
    if args.tat:
        list_word_pairs = get_tat_datas(args.input_dir)
    else:
        sents = get_all_datas(args.input_dir, args.prefixes)
        list_word_pairs = list(map(parse_sent, sents))
    before_trimming = len(list_word_pairs)
    list_word_pairs = list(filter(lambda word_pairs: len(word_pairs) > 0, list_word_pairs))
    logging.warning(f"dumped {before_trimming - len(list_word_pairs)} out of {len(list_word_pairs)} examples")
    #if args.vocab_path is not None:
    #    counter = Counter(flatten(map(lambda sent: sent.split(), src_sents)))
    #    write_lines_to_file([word for word, count in counter.most_common()], args.vocab_path)
    write_lines_to_file(list_word_pairs,
                        f"{args.output_dir}/all.txt")

parser = argparse.ArgumentParser()
parser.add_argument('input_dir')
parser.add_argument('output_dir')
parser.add_argument('--prefixes', nargs='+')
parser.add_argument('--tat', action='store_true')
args = parser.parse_args()
main(args)
