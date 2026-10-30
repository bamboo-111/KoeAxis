"""Offline gold ASS diagnostics. No model, network, or index-based cue pairing."""
import argparse
import difflib
import hashlib
import importlib.util
import json
import re
import statistics
import unicodedata
from pathlib import Path


def norm(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text) if c.isalnum())


def clean(text):
    return re.sub(r'\{[^}]*\}', '', text).replace('\\N', ' ').replace('\\n', ' ')


def time_ms(text):
    h, m, s = text.split(':')
    return round((int(h)*3600 + int(m)*60 + float(s))*1000)


def speaker(style):
    return next((x for x in ('理名', '夕莉', '朱李') if x in style), style)


def read_ass(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.startswith('Dialogue:'):
            continue
        f = line.split(':', 1)[1].strip().split(',', 9)
        if len(f) != 10:
            raise ValueError('invalid ASS Dialogue at line %s' % line_no)
        text = clean(f[9])
        rows.append({'i': line_no, 's': time_ms(f[1]), 'e': time_ms(f[2]), 'speaker': speaker(f[3]), 'note': f[4], 'ja': text, 'empty': not text.strip(), 'question': bool(re.search(r'[?？]{2,}', text)), 'multi': '一起' in f[4]})
    return rows


def read_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    data = list(data.values()) if isinstance(data, dict) else data
    return [{'i': i, 's': x['start_time'], 'e': x['end_time'], 'speaker': x.get('speaker') or '', 'ja': x.get('original_subtitle') or '', 'overlay': bool(x.get('overlay'))} for i, x in enumerate(data)]


def overlap(a, b):
    return max(0, min(a['e'], b['e']) - max(a['s'], b['s']))


def unknown(r):
    return r['empty'] or r['question'] or not norm(r['ja'])


def safe_reference(rows):
    bad = [r for r in rows if unknown(r)]
    return [r for r in rows if not unknown(r) and not any(overlap(r, u) for u in bad)]


def edits(a, b):
    # Levenshtein with deterministic tie preference: substitution, deletion, insertion.
    prev = [(j, 0, 0, j) for j in range(len(b)+1)]
    for i, x in enumerate(a, 1):
        cur = [(i, 0, i, 0)]
        for j, y in enumerate(b, 1):
            d = prev[j-1]; sub = (d[0] + (x != y), d[1] + (x != y), d[2], d[3])
            d = prev[j]; delete = (d[0]+1, d[1], d[2]+1, d[3])
            d = cur[j-1]; insert = (d[0]+1, d[1], d[2], d[3]+1)
            cur.append(min((sub, delete, insert), key=lambda z:z[0]))
        prev = cur
    return prev[-1]


def content(rows, candidates, strict=False):
    # A candidate touching multiple reference rows or unknown/overlap is ambiguous;
    # exclude those rows instead of duplicating text or scoring across unknown spans.
    safe = safe_reference(rows)
    selected = []
    for r in safe:
        if any(overlap(r, x) for x in rows if x is not r):
            continue
        cs = [c for c in candidates if overlap(r, c)]
        if any(sum(bool(overlap(c, x)) for x in rows) != 1 for c in cs):
            continue
        if strict and any(overlap(a, b) for i, a in enumerate(cs) for b in cs[i+1:]):
            continue
        selected.append((r, sorted(cs, key=lambda x:(x['s'], x['e']))))
    totals = {'reference_rows': len(selected), 'reference_chars': 0, 'candidate_chars': 0, 'edits': 0, 'substitutions': 0, 'deletions': 0, 'insertions': 0}
    examples = []
    row_scores = []
    for r, cs in selected:
        a = norm(r['ja']); b = ''.join(norm(c['ja']) for c in cs)
        distance, sub, delete, insert = edits(a, b)
        totals['reference_chars'] += len(a); totals['candidate_chars'] += len(b)
        for key, value in [('edits', distance), ('substitutions', sub), ('deletions', delete), ('insertions', insert)]: totals[key] += value
        row_scores.append({'reference_line':r['i'], 'reference_chars':len(a), 'edits':distance, 'reference':r['ja'], 'candidate':' '.join(c['ja'] for c in cs)})
        if distance: examples.append({'reference_line':r['i'], 'start_ms':r['s'], 'reference':r['ja'], 'candidate':' '.join(c['ja'] for c in cs), 'edits':distance})
    totals['examples'] = sorted(examples, key=lambda x:x['edits'], reverse=True)[:25]
    totals['row_scores'] = row_scores
    return totals


def speaker_time(rows, candidates):
    points = sorted({p for r in rows+candidates for p in (r['s'], r['e'])})
    durations = dict.fromkeys(('same', 'wrong', 'unlabeled', 'uncovered', 'candidate_multiple'), 0)
    excluded = {'unknown_ms': 0, 'reference_overlap_ms': 0, 'multi_ms': 0}
    for s, e in zip(points, points[1:]):
        mid = (s+e)/2
        rr = [r for r in rows if r['s'] <= mid < r['e']]
        if not rr: continue
        if len(rr) != 1: excluded['reference_overlap_ms'] += e-s; continue
        r = rr[0]
        if unknown(r): excluded['unknown_ms'] += e-s; continue
        if r['multi']: excluded['multi_ms'] += e-s; continue
        cs = [c for c in candidates if c['s'] <= mid < c['e']]
        labels = {c['speaker'] for c in cs if c['speaker']}
        key = 'uncovered' if not cs else 'unlabeled' if not labels else 'candidate_multiple' if len(labels)>1 else 'same' if labels == {r['speaker']} else 'wrong'
        durations[key] += e-s
    return {'eligible_ms': sum(durations.values()), 'durations_ms': durations, 'excluded_ms': excluded}


def timings(rows, candidates, strict=False):
    pairs = []
    for r in safe_reference(rows):
        if len(norm(r['ja'])) < 4: continue
        cs = [c for c in candidates if norm(c['ja']) == norm(r['ja']) and overlap(r, c)]
        if strict:
            cs = [c for c in cs if sum(bool(overlap(c, x)) for x in rows) == 1]
            if len(cs) != 1: continue
        if cs:
            c = max(cs, key=lambda c:overlap(r, c))
            pairs.append({'ref_line':r['i'], 'start_delta_ms':c['s']-r['s'], 'end_delta_ms':c['e']-r['e']})
    def stats(key):
        a = [p[key] for p in pairs]
        return {'n':len(a), 'median_signed_ms':statistics.median(a) if a else None, 'median_absolute_ms':statistics.median(map(abs,a)) if a else None}
    return {'pairs':pairs, 'start':stats('start_delta_ms'), 'end':stats('end_delta_ms')}


def token_segmentation(rows, candidates, token_path, legacy_path):
    spec = importlib.util.spec_from_file_location('gold54_legacy_score', legacy_path)
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    gold = [{'s':r['s']/1000, 'e':r['e']/1000, 'spk':r['speaker'], 'n':legacy.norm(r['ja']), 'text':r['ja'], 'ref':r} for r in rows]
    toks = legacy.load_tokens(Path(token_path), 0, max((r['e'] for r in rows), default=0)/1000)
    legacy.assign_tokens(toks, gold)
    for t in toks:
        r = gold[t['g']]['ref'] if t['g'] is not None else None
        t['eligible'] = bool(r and not unknown(r) and not r['multi'] and not any(overlap(r,u) for u in rows if u is not r))
    cues = sorted([{'s':c['s']/1000, 'e':c['e']/1000, 'text':c['ja'], 'overlay':c.get('overlay', False)} for c in candidates], key=lambda c:(c['s'],c['e']))
    owner = legacy.map_cues(toks, cues)
    counts = dict.fromkeys(('within_unit_pairs','wrong_breaks','different_unit_pairs','missed_ref_breaks','speaker_change_pairs','missed_speaker_breaks'),0)
    for i, (a,b) in enumerate(zip(toks,toks[1:])):
        if not a['eligible'] or not b['eligible'] or owner[i] is None or owner[i+1] is None: continue
        br = owner[i] != owner[i+1]
        if a['g'] == b['g']:
            counts['within_unit_pairs'] += 1; counts['wrong_breaks'] += int(br)
        else:
            counts['different_unit_pairs'] += 1; counts['missed_ref_breaks'] += int(not br)
            if gold[a['g']]['spk'] != gold[b['g']]['spk']:
                counts['speaker_change_pairs'] += 1; counts['missed_speaker_breaks'] += int(not br)
    return {'tokens':len(toks), 'eligible_tokens':sum(t['eligible'] for t in toks), 'eligible_unmapped_tokens':sum(t['eligible'] and owner[i] is None for i,t in enumerate(toks)), 'counts':counts}


def legacy_quality(rows, candidates, reference, legacy_root):
    # Import the unchanged old implementation, not a recreated approximation.
    import sys
    root = str(Path(legacy_root).resolve())
    sys.path.insert(0, root)
    try:
        from qwen_asr.ass_quality import evaluate_ass_quality, SubtitleCue
        from qwen_asr.history_glossary import AssDialogue
        import qwen_asr.ass_quality as quality
        if not Path(quality.__file__).resolve().is_relative_to(Path(root)):
            raise ValueError('loaded qwen_asr does not belong to requested legacy root')
        dialogues = [AssDialogue(start_ms=r['s'], end_ms=r['e'], style=r['speaker'], text=r['ja']) for r in sorted(safe_reference(rows), key=lambda x:(x['s'],x['e']))]
        cues = [SubtitleCue(text=c['ja'], start_ms=c['s'], end_ms=c['e'], key=str(c['i'])) for c in candidates]
        return evaluate_ass_quality(ass_path=Path(reference), source='explicit candidate', dialogues=dialogues, cues=cues, offset_ms=0, window_ms=1200, diagnostic_window_ms=8000, low_score_threshold=.45, fail_score_threshold=.20, max_cases=25)
    finally:
        sys.path.remove(root)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(reference, split, normalized, ass, tokens=None, legacy=None, legacy_root=None):
    rows = read_ass(reference)
    stages = {'split':read_manifest(split), 'normalized':read_manifest(normalized), 'ass':read_ass(ass)}
    out = {'inputs':{str(p):digest(p) for p in (reference,split,normalized,ass)}, 'reference':{'all':len(rows), 'empty':sum(r['empty'] for r in rows), 'question_rows':sum(r['question'] for r in rows), 'safe':len(safe_reference(rows)), 'multi':sum(r['multi'] for r in rows)}, 'stages':{}, 'chinese':'NOT_ASSESSED: no complete human Chinese gold'}
    for stage in ('split','normalized'):
        cs = stages[stage]
        result = {'count':len(cs), 'content':content(rows,cs), 'time':timings(rows,cs), 'speaker':speaker_time(rows,cs)}
        if tokens and legacy: result['token_segmentation'] = token_segmentation(rows,cs,tokens,legacy)
        else: result['token_segmentation'] = 'NOT_ASSESSED: provide --tokens and --legacy-score'
        result['legacy_ass_match'] = legacy_quality(rows, cs, reference, legacy_root) if legacy_root else 'NOT_ASSESSED: provide --legacy-root'
        out['stages'][stage] = result
    out['ass_speaker'] = speaker_time(rows, stages['ass'])
    if tokens and legacy:
        out['inputs'][str(tokens)] = digest(tokens); out['inputs'][str(legacy)] = digest(legacy)
    if legacy_root:
        for name in ('ass_quality.py', 'history_glossary.py'):
            path = Path(legacy_root) / 'qwen_asr' / name
            out['inputs'][str(path)] = digest(path)
    return out


def evaluate_v2(reference, split, normalized, ass, tokens=None, legacy=None, legacy_root=None):
    """Conservative cue-level diagnostic; no quality gate or substring verdict."""
    result = evaluate(reference, split, normalized, ass, tokens, legacy, legacy_root)
    rows = read_ass(reference)
    # Keep excluded rows in the timeline so a cue crossing them remains ambiguous.
    strict = [dict(r, question=True) if r['multi'] or any(overlap(r, x) for x in rows if x is not r) else r for r in rows]
    for stage, path in (('split', split), ('normalized', normalized)):
        cs = read_manifest(path)
        result['stages'][stage]['content'] = content(strict, cs, strict=True)
        result['stages'][stage]['time'] = timings(strict, cs, strict=True)
    result['schema'] = 'gold54-human-reference-diagnostic/v2'
    result['verdict'] = 'NOT_ASSESSED: diagnostic only; no acceptance threshold'
    result['axes'] = {'text': 'isolated reference rows versus overlapping candidate cues',
                      'pronunciation_time': 'split identical-text overlapping pairs only',
                      'display_time': 'normalized identical-text overlapping pairs only',
                      'segmentation': 'common ASR tokens when provided; otherwise NOT_ASSESSED',
                      'speaker': 'eligible reference time, excluding unknown, multi and overlap',
                      'legacy_ass_match': 'optional legacy similarity against supplied cues, not final subtitle accuracy'}
    result['limitations'] = 'Unknown, multi and overlapping references excluded from text and timing; cross-reference cues excluded from text; no substring correctness inference.'
    return result


def worsening(baseline, candidate):
    result = {}
    for stage in ('split','normalized'):
        b, c = baseline['stages'][stage], candidate['stages'][stage]
        bc = {r['reference_line']:r for r in b['content']['row_scores']}
        cc = {r['reference_line']:r for r in c['content']['row_scores']}
        common = sorted(bc.keys() & cc.keys())
        result[stage] = {'common_rows':len(common), 'common_reference_chars':sum(bc[i]['reference_chars'] for i in common), 'baseline_edits':sum(bc[i]['edits'] for i in common), 'candidate_edits':sum(cc[i]['edits'] for i in common), 'worsened_rows':sum(cc[i]['edits'] > bc[i]['edits'] for i in common), 'improved_rows':sum(cc[i]['edits'] < bc[i]['edits'] for i in common), 'worsened_examples':[{'reference_line':i,'reference':bc[i]['reference'],'baseline':bc[i]['candidate'],'candidate':cc[i]['candidate'],'baseline_edits':bc[i]['edits'],'candidate_edits':cc[i]['edits']} for i in common if cc[i]['edits'] > bc[i]['edits']][:25]}
        result[stage]['content_edit_delta'] = result[stage]['candidate_edits'] - result[stage]['baseline_edits']
        if isinstance(b['token_segmentation'],dict) and isinstance(c['token_segmentation'],dict):
            result[stage]['token_counts_delta'] = {k:c['token_segmentation']['counts'][k]-v for k,v in b['token_segmentation']['counts'].items()}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',required=True); ap.add_argument('--candidate',nargs=4,action='append',metavar=('LABEL','SPLIT','NORMALIZED','ASS'),required=True)
    ap.add_argument('--tokens'); ap.add_argument('--legacy-score'); ap.add_argument('--legacy-root'); ap.add_argument('--baseline'); ap.add_argument('--output',required=True)
    ap.add_argument('--schema', choices=('legacy', 'gold54-human-reference-diagnostic/v2'), default='legacy')
    args = ap.parse_args()
    if bool(args.tokens) != bool(args.legacy_score): ap.error('--tokens and --legacy-score must be provided together')
    evaluator = evaluate_v2 if args.schema != 'legacy' else evaluate
    results = {label:evaluator(args.reference,s,n,a,args.tokens,args.legacy_score,args.legacy_root) for label,s,n,a in args.candidate}
    if args.baseline and args.baseline not in results: ap.error('unknown baseline')
    report = {'method':'NFKC + alphanumeric; per clear isolated reference row, only candidate cues touching exactly that row; no unknown-span concatenation. Time matching requires overlapping identical normalized text >=4 chars. ASS Style supplies speaker. Matching is diagnostic, not accuracy.', 'candidates':results}
    if args.schema != 'legacy': report['schema'] = args.schema; report['verdict'] = 'NOT_ASSESSED: diagnostic only'
    if args.schema != 'legacy': report['legacy_aligned'] = 'NOT_ASSESSED: separately retain original source=aligned report; not final subtitle quality'
    if args.schema != 'legacy': report['common_denominator'] = 'worsening comparison intersects eligible reference lines per stage'
    if args.schema != 'legacy': report['token_delta_warning'] = 'token pair counts are diagnostics, not automatically a common denominator'
    if args.baseline: report['worsening_vs_'+args.baseline] = {k:worsening(results[args.baseline],v) for k,v in results.items() if k != args.baseline}
    out = Path(args.output)
    if out.exists(): raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x',encoding='utf-8',newline='\n') as f: json.dump(report,f,ensure_ascii=False,indent=2); f.write('\n')


if __name__ == '__main__': main()
