import os, sys
sys.modules['flash_attn'] = None
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')

import argparse, datetime, hashlib, json, math, shutil, subprocess, time
from collections import defaultdict

NL = chr(10)
DOMAIN = 'liver'
ALL_MODELS = ['bge-base', 'qwen3-0.6b']
ALL_METHODS = ['mnrl', 'bm25', 'dense', 'dense_rank10', 'dense_rank100',
               'random', 'margin', 'margin_onto',
               'onto_pool_sim', 'onto_pool_rand',
               'onto_pool_strat', 'onto_strat_loose', 'onto_loose_rand',
               'onto_pool_hard', 'onto_hard_loose',
               'onto_pool_mid', 'onto_nmid_loose', 'onto_hmid_loose',
               'onto_dose_p10', 'onto_dose_p30', 'onto_dose_p50',
               'bm25_onto', 'dense_onto']

# 假负例剂量实验（--tag new_dose）：安全池里按比例混入同病答案；0% 即 onto_pool_rand
DOSE_LEVELS = [('onto_pool_rand', 0), ('onto_dose_p10', 10), ('onto_dose_p30', 30), ('onto_dose_p50', 50)]

# 主要比较
PRIMARY_COMPARE = [
    # 剂量实验（--tag new_dose）：逐档比较，以及对 random / mnrl
    ('onto_dose_p10',    'onto_pool_rand'),
    ('onto_dose_p30',    'onto_pool_rand'),
    ('onto_dose_p50',    'onto_pool_rand'),
    ('onto_dose_p30',    'onto_dose_p10'),
    ('onto_dose_p50',    'onto_dose_p30'),
    ('onto_dose_p30',    'random'),
    ('onto_dose_p50',    'mnrl'),
    # 主线（--tag new）：本体边界有没有用
    ('onto_pool_rand',   'mnrl'),
    ('onto_loose_rand',  'mnrl'),
    ('onto_loose_rand',  'random'),
    ('onto_pool_rand',   'random'),
    ('onto_pool_rand',   'onto_loose_rand'),
    ('onto_pool_strat',  'onto_pool_rand'),
    ('onto_strat_loose', 'onto_loose_rand'),
    ('onto_pool_rand',   'margin'),
    ('onto_loose_rand',  'margin'),
    # 打分类（--tag new_core）：池内打分挑难
    # 核心：宽松池故意选最难 vs random（有27%假负例的随机）
    ('onto_hard_loose',  'random'),
    # 核心：零模型难度选法 vs 模型难度选法
    ('onto_hard_loose',  'margin'),
    # 安全池加难度 vs 安全池随机
    ('onto_pool_hard',   'onto_pool_rand'),
    # 池边界影响：安全 vs 宽松（同打分器）
    ('onto_pool_hard',   'onto_hard_loose'),
    # 打分类 vs mnrl
    ('onto_hard_loose',  'mnrl'),
]

# 探索性比较
EXPLORATORY_COMPARE = [
    ('onto_pool_sim',         'onto_pool_rand'),  # 实体余弦排序有没有用
    ('onto_pool_sim',         'random'),
    ('onto_pool_sim',         'mnrl'),
    ('onto_pool_mid',         'onto_pool_rand'),
    ('onto_nmid_loose',       'onto_loose_rand'),
    ('onto_hmid_loose',       'onto_loose_rand'),
    ('margin',              'mnrl'),
    ('margin',              'random'),
    ('margin_onto',         'margin'),
    ('margin_onto',         'mnrl'),
    # 本体过滤对 bm25 / dense 候选的作用（--tag new_core，与 margin_onto 同口径）
    ('bm25_onto',           'bm25'),
    ('dense_onto',          'dense'),
]
COMPARE = PRIMARY_COMPARE + EXPLORATORY_COMPARE

# =============================================================================
# 配置
# =============================================================================

def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir',    default=os.environ.get('LIVER_OUT_DIR', '/home/liver/data/liver_dataset'))
    ap.add_argument('--model-root', default=os.environ.get('LIVER_MODEL_ROOT', '/data/finetuned_models'))
    ap.add_argument('--tag',        default='new')
    ap.add_argument('--model',      choices=ALL_MODELS)
    ap.add_argument('--method',     choices=ALL_METHODS + ['zeroshot'])
    ap.add_argument('--seed',       type=int, default=42)
    ap.add_argument('--max-steps',  type=int, default=None)
    ap.add_argument('--overwrite',  action='store_true')
    ap.add_argument('--force',      action='store_true', help='--launch 时忽略已有结果：删掉对应的结果目录和模型目录后重训')
    ap.add_argument('--launch',     action='store_true')
    ap.add_argument('--report',     action='store_true')
    ap.add_argument('--models',     default=','.join(ALL_MODELS))
    ap.add_argument('--methods',    default=','.join(ALL_METHODS))
    ap.add_argument('--seeds',      default='42,43,44')
    ap.add_argument('--gpus',       default='0,1,2,3')
    ap.add_argument('--epochs',     type=int, default=1)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--symmetric',  action='store_true')
    ap.add_argument('--no-reuse-old', dest='reuse_old', action='store_false')
    a, _ = ap.parse_known_args()
    return a

ARGS       = _args()
OUT_DIR    = ARGS.out_dir
TAG        = ARGS.tag
SPLIT_DIR  = f'{OUT_DIR}/split'
TRAIN_DIR  = f'{OUT_DIR}/training_{TAG}'
OLD_TRAIN  = f'{OUT_DIR}/training'
OLD_RES    = f'{OUT_DIR}/results'
REPORTS    = f'{OUT_DIR}/reports'
MODEL_ROOT = ARGS.model_root
CKPT_ROOT  = f'{MODEL_ROOT}/_checkpoints'
SCRIPT     = os.path.abspath(globals().get('__file__', '09_train_eval_new.py'))
FORCE      = ARGS.force

MODELS = {
    'bge-base': {
        'path':         '/data/bge-base-zh-v1.5',
        'query_prompt': '为这个句子生成表示以用于检索相关文章：',
        'lr':           2e-5,
        'mini_batch':   32,
        'eval_dtype':   'float16',
        'eval_batch':   128,
        'max_parallel': 4,
    },
    'qwen3-0.6b': {
        'path':         '/data/Qwen3-Embedding-0.6B',
        'query_prompt': 'Instruct: Given a web search query, retrieve relevant passages that answer the query' + NL + 'Query:',
        'lr':           1e-5,
        'mini_batch':   16,
        'eval_dtype':   'bfloat16',
        'eval_batch':   32,
    },
}

MODEL_LIST = [m for m in ARGS.models.split(',') if m in MODELS]
METHODS    = [m for m in ARGS.methods.split(',') if m in ALL_METHODS]
SEEDS      = [int(s) for s in ARGS.seeds.split(',') if s]
GPUS       = [int(g) for g in ARGS.gpus.split(',') if g != '']

BATCH_SIZE  = ARGS.batch_size
EPOCHS      = ARGS.epochs
SYMMETRIC   = ARGS.symmetric
LOSS_NAME   = ('CachedMultipleNegativesSymmetricRankingLoss' if SYMMETRIC
               else 'CachedMultipleNegativesRankingLoss')
WARMUP      = 0.1
MAX_SEQ     = 512
EVAL_STEPS  = 100
TOP_K       = 100
PERM_ITERS  = 10000
BOOT_ITERS  = 2000
NONINF_MARGIN = 0.30   # 非劣效界（nDCG@10 百分点）：差值 95% CI 下界 > -0.30 即判定不劣于基线

SUFFIX = ''
if (BATCH_SIZE, EPOCHS, SYMMETRIC) != (64, 1, False):
    SUFFIX = '_ep{}_b{}'.format(EPOCHS, BATCH_SIZE) + ('_sym' if SYMMETRIC else '')
RUN_TAG = TAG + SUFFIX
RESULTS = f'{OUT_DIR}/results_{RUN_TAG}'

IDLE_MB    = 2000
MAX_TRIES  = 3
START_WAIT = 60

# 所有 onto_* 方法共用 bge 的训练文件（不依赖模型）
SHARED_METHODS = {'onto_pool_sim', 'onto_pool_rand',
                  'onto_pool_hard', 'onto_hard_loose',
                  'onto_pool_strat', 'onto_strat_loose', 'onto_loose_rand',
                  'onto_pool_mid', 'onto_nmid_loose', 'onto_hmid_loose',
                  'onto_dose_p10', 'onto_dose_p30', 'onto_dose_p50',
                  'random', 'mnrl'}

# =============================================================================
# 路径与公共函数
# =============================================================================

def run_name(model, method, seed):
    return '{}_{}_{}_seed{}_{}'.format(DOMAIN, model, method, seed, RUN_TAG)

def _dir(root, model, method, seed, smoke=False):
    if smoke:       return f'{root}/_smoke/{model}/{method}/seed{seed}'
    if method == 'zeroshot': return f'{root}/{model}/zeroshot'
    return f'{root}/{model}/{method}/seed{seed}'

def run_dir(model, method, seed, smoke=False): return _dir(RESULTS, model, method, seed, smoke)
def old_run_dir(model, method, seed):          return _dir(OLD_RES, model, method, seed)
def model_dir(model, method, seed):            return f'{MODEL_ROOT}/{run_name(model, method, seed)}'

def read_jsonl(p):
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def load_json(p):
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def file_md5(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_ir(split):
    queries = read_jsonl(f'{SPLIT_DIR}/{split}_queries.jsonl')
    corpus  = read_jsonl(f'{SPLIT_DIR}/{split}_corpus.jsonl')
    rel     = defaultdict(set)
    for r in read_jsonl(f'{SPLIT_DIR}/{split}_qrels.jsonl'):
        rel[r['qid']].add(r['doc_id'])
    return queries, corpus, rel

def query_metrics(ranked, rel):
    dcg  = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in rel)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(rel), 10)))
    rr   = next((1 / (i + 1) for i, d in enumerate(ranked[:10]) if d in rel), 0.0)
    m    = {'ndcg@10': dcg / idcg, 'mrr@10': rr}
    for k in (1, 5, 10, 20, 100):
        m['recall@{}'.format(k)] = len(rel & set(ranked[:k])) / len(rel)
    return m

METRICS = ['ndcg@10', 'mrr@10', 'recall@1', 'recall@5', 'recall@10', 'recall@20', 'recall@100']

def recipe_config(model, method):
    c   = MODELS[model]
    cfg = {'query_prompt': c['query_prompt'], 'max_seq_len': MAX_SEQ, 'eval_dtype': c['eval_dtype']}
    if method != 'zeroshot':
        cfg.update({'lr': c['lr'], 'batch_size': BATCH_SIZE, 'epochs': EPOCHS,
                    'warmup_ratio': WARMUP, 'eval_steps': EVAL_STEPS, 'loss': LOSS_NAME})
    return cfg

def _train_file(model, method):
    '''onto_* (all ontology methods) / random / mnrl 与模型无关，共用 bge-base 的文件'''
    ref = 'bge-base' if method in SHARED_METHODS else model
    return f'{TRAIN_DIR}/{ref}/{method}.jsonl'

def try_reuse(model, method, seed):
    if not ARGS.reuse_old or SUFFIX:
        return False
    src = old_run_dir(model, method, seed)
    mp  = f'{src}/metrics.json'
    if not (os.path.exists(mp) and os.path.exists(f'{src}/per_query.jsonl')):
        return False
    old = load_json(mp)
    if any(old.get('config', {}).get(k) != v for k, v in recipe_config(model, method).items()):
        return False
    if method != 'zeroshot':
        nf = _train_file(model, method)
        of = f'{OLD_TRAIN}/{model}/{method}.jsonl'   # old files were model-specific
        if not (os.path.exists(nf) and os.path.exists(of)) or file_md5(nf) != file_md5(of):
            return False
    dst = run_dir(model, method, seed)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    old['reused_from'] = mp
    with open(f'{dst}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(old, f, ensure_ascii=False, indent=2)
    return True

def evaluate_test(model, prompt, out_dir, eval_batch):
    import torch
    queries, corpus, rel = load_ir('test')
    doc_ids = [r['doc_id'] for r in corpus]
    d_emb   = model.encode([r['text'] for r in corpus], batch_size=eval_batch,
                            normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    q_emb   = model.encode([r['text'] for r in queries], prompt=prompt, batch_size=eval_batch,
                            normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
    per_q = []
    with torch.no_grad():
        for s in range(0, len(queries), 1024):
            sims = (q_emb[s:s + 1024] @ d_emb.T).float()
            idx  = torch.topk(sims, k=TOP_K, dim=1).indices.cpu().tolist()
            for j, row in enumerate(idx):
                q      = queries[s + j]
                ranked = [doc_ids[i] for i in row]
                m      = query_metrics(ranked, rel[q['qid']])
                per_q.append({'qid': q['qid'], 'source': q.get('source', ''), **m, 'top10': ranked[:10]})
    with open(f'{out_dir}/per_query.jsonl', 'w', encoding='utf-8') as f:
        for r in per_q:
            f.write(json.dumps(r, ensure_ascii=False) + NL)
    overall   = {k: sum(r[k] for r in per_q) / len(per_q) for k in METRICS}
    by_src    = defaultdict(list)
    for r in per_q:
        by_src[r['source']].append(r)
    by_source = {s: {'n': len(rs), **{k: sum(r[k] for r in rs) / len(rs) for k in METRICS}}
                 for s, rs in by_src.items()}
    return overall, by_source

# =============================================================================
# 单次运行
# =============================================================================

def run(model_key, method, seed, max_steps=None, overwrite=False):
    smoke = max_steps is not None
    if not smoke and try_reuse(model_key, method, seed):
        print('reused  {} {} seed{} -> {}'.format(model_key, method, seed,
                                                   run_dir(model_key, method, seed)))
        return

    import torch
    from sentence_transformers import SentenceTransformer

    cfg        = MODELS[model_key]
    eval_batch = cfg['eval_batch']
    out_dir    = run_dir(model_key, method, seed, smoke)
    os.makedirs(out_dir, exist_ok=True)
    if not smoke and os.path.exists(f'{out_dir}/metrics.json'):
        os.remove(f'{out_dir}/metrics.json')    
    t0   = time.time()
    info = {'model': model_key, 'method': method, 'seed': seed, 'domain': DOMAIN, 'tag': RUN_TAG,
            'started_at': datetime.datetime.now().isoformat(timespec='seconds'),
            'gpu': os.environ.get('CUDA_VISIBLE_DEVICES', ''),
            'config': {**recipe_config(model_key, method),
                       'base_model_path': cfg['path'],
                       'mini_batch': cfg['mini_batch'], 'batch_sampler': 'NO_DUPLICATES',
                       'symmetric': SYMMETRIC, 'eval_batch': eval_batch}}

    # ---- 零样本 ----
    if method == 'zeroshot':
        model = SentenceTransformer(cfg['path'], device='cuda', local_files_only=True,
                                    model_kwargs={'torch_dtype': getattr(torch, cfg['eval_dtype'])})
        model.max_seq_length = MAX_SEQ
        overall, by_source = evaluate_test(model, cfg['query_prompt'], out_dir, eval_batch)
        info.update({'model_path': cfg['path'], 'test': overall, 'test_by_source': by_source,
                     'runtime_min': round((time.time() - t0) / 60, 1)})
        with open(f'{out_dir}/metrics.json', 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        print(json.dumps(overall, indent=2))
        return

    # ---- 防覆盖 ----
    save_to = model_dir(model_key, method, seed)
    if not smoke and os.path.exists(save_to):
        if not overwrite:
            raise SystemExit(save_to + ' exists. Add --overwrite to replace.')
        shutil.rmtree(save_to)

    # ---- 训练 ----
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.evaluation import InformationRetrievalEvaluator
    from sentence_transformers.training_args import BatchSamplers

    train_file = _train_file(model_key, method)
    assert os.path.exists(train_file), train_file + ' not found — run 08_training_sets_new.py first'
    cols  = ['anchor', 'positive'] + ([] if method == 'mnrl' else ['negative'])
    train = load_dataset('json', data_files=train_file, split='train').select_columns(cols).shuffle(seed=seed)  # NO_DUPLICATES 采样器自带 seed=0，不受 data_seed 控制，先按种子打乱数据行

    model = SentenceTransformer(cfg['path'], device='cuda', local_files_only=True)
    model.max_seq_length = MAX_SEQ
    if SYMMETRIC:
        from sentence_transformers.losses import CachedMultipleNegativesSymmetricRankingLoss
        loss = CachedMultipleNegativesSymmetricRankingLoss(model, mini_batch_size=cfg['mini_batch'])
    else:
        from sentence_transformers.losses import CachedMultipleNegativesRankingLoss
        loss = CachedMultipleNegativesRankingLoss(model, mini_batch_size=cfg['mini_batch'])

    dq, dc, drel = load_ir('dev')
    dev_eval = InformationRetrievalEvaluator(
        queries={r['qid']: r['text'] for r in dq},
        corpus={r['doc_id']: r['text'] for r in dc},
        relevant_docs=dict(drel), name='dev', query_prompt=cfg['query_prompt'],
        ndcg_at_k=[10], mrr_at_k=[10], accuracy_at_k=[1, 10],
        precision_recall_at_k=[10, 100], map_at_k=[100],
        batch_size=eval_batch, show_progress_bar=False)

    name     = run_name(model_key, method, seed) + ('_smoke' if smoke else '')
    ckpt_dir = f'{CKPT_ROOT}/{name}'
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    steps = max_steps or EVAL_STEPS
    args  = SentenceTransformerTrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=EPOCHS,
        max_steps=max_steps or -1,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=cfg['lr'],
        warmup_ratio=WARMUP,
        lr_scheduler_type='linear',
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        prompts={'anchor': cfg['query_prompt']},
        eval_strategy='steps', eval_steps=steps,
        save_strategy='steps', save_steps=steps, save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='eval_dev_cosine_ndcg@10', greater_is_better=True,
        logging_steps=25, seed=seed, data_seed=seed,
        report_to='none', dataloader_num_workers=2,
    )
    trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=train,
                                          loss=loss, evaluator=dev_eval)
    trainer.train()
    model = trainer.model
    info['train'] = {
        'train_file': train_file, 'train_file_md5': file_md5(train_file), 'rows': len(train),
        'shared_file': method in SHARED_METHODS,
        'steps': trainer.state.global_step,
        'best_dev_ndcg@10': trainer.state.best_metric,
        'best_checkpoint': os.path.basename(trainer.state.best_model_checkpoint or ''),
        'peak_gpu_gb': round(torch.cuda.max_memory_allocated() / 1e9, 1),
        'train_min': round((time.time() - t0) / 60, 1),
    }
    with open(f'{out_dir}/train_log.json', 'w', encoding='utf-8') as f:
        json.dump(trainer.state.log_history, f, indent=1)

    if smoke:
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        sps  = (time.time() - t0) / max_steps
        full = math.ceil(len(train) / BATCH_SIZE) * EPOCHS
        print(NL + 'SMOKE OK  peak {} GB  ~{:.1f}s/step  full ~{:.0f}min'.format(
            info['train']['peak_gpu_gb'], sps, full * sps / 60))
        return

    del trainer
    torch.cuda.empty_cache()
    model.save_pretrained(save_to)
    shutil.rmtree(ckpt_dir, ignore_errors=True)
    info['model_path'] = save_to

    model.to(getattr(torch, cfg['eval_dtype']))
    torch.cuda.empty_cache()
    overall, by_source = evaluate_test(model, cfg['query_prompt'], out_dir, eval_batch)
    info.update({'test': overall, 'test_by_source': by_source,
                 'runtime_min': round((time.time() - t0) / 60, 1)})

    with open(f'{save_to}/training_info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    with open(f'{out_dir}/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(overall, indent=2))
    print(NL + 'Model saved to ' + save_to)

# =============================================================================
# 调度
# =============================================================================

def gpu_used_mb():
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=30).stdout
        return {int(i): int(u) for i, u in (l.split(',') for l in out.strip().splitlines())}
    except Exception:
        return {}

def now(): return datetime.datetime.now().strftime('%m-%d %H:%M')

def launch():
    jobs = [(m, 'zeroshot', 0) for m in MODEL_LIST]
    for m in MODEL_LIST:
        for s in SEEDS:
            for meth in METHODS:
                jobs.append((m, meth, s))

    pending = []
    for j in jobs:
        if FORCE:
            shutil.rmtree(run_dir(*j), ignore_errors=True)
            if j[1] != 'zeroshot':
                shutil.rmtree(model_dir(*j), ignore_errors=True)
            pending.append(j)
            continue
        if os.path.exists(f'{run_dir(*j)}/metrics.json'):
            continue
        if try_reuse(*j):
            print('  reused  {} {} seed{}'.format(*j), flush=True); continue
        if j[1] != 'zeroshot' and os.path.exists(model_dir(*j)):
            shutil.rmtree(model_dir(*j))
        pending.append(j)

    os.makedirs(f'{RESULTS}/logs', exist_ok=True)
    os.makedirs(MODEL_ROOT, exist_ok=True)
    n_total = len(jobs)
    n_todo  = len(pending)
    n_done  = n_total - n_todo
    print('{} runs total  |  {} already done  |  {} to do  |  GPUs {}  |  tag {}'.format(
        n_total, n_done, n_todo, GPUS, RUN_TAG), flush=True)
    print('-' * 70, flush=True)

    tries, free, running = defaultdict(int), list(GPUS), {}
    last_wait = 0
    completed_count = n_done
    while pending or running:
        used = gpu_used_mb()
        for g in list(free):
            if not pending: break
            if used.get(g, 0) >= IDLE_MB: continue
            job  = pending[0]
            cap  = MODELS[job[0]].get('max_parallel', len(GPUS))
            if sum(1 for v in running.values() if v[1][0] == job[0]) >= cap: break
            free.remove(g); pending.pop(0)
            m, meth, s = job
            tries[job] += 1
            name = '{}_{}_seed{}'.format(m, meth, s)
            log  = open(f'{RESULTS}/logs/{name}.log', 'a' if tries[job] > 1 else 'w')
            cmd  = [sys.executable, SCRIPT,
                    '--model', m, '--method', meth, '--seed', str(s),
                    '--out-dir', OUT_DIR, '--model-root', MODEL_ROOT, '--tag', TAG,
                    '--epochs', str(EPOCHS), '--batch-size', str(BATCH_SIZE), '--no-reuse-old']
            if SYMMETRIC: cmd.append('--symmetric')
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(g)))
            running[p] = (g, job, name, log, time.time())
            pct = 100.0 * completed_count / n_total
            print('[{}] START  {:<38s} GPU {}  (try {})  [{}/{}  {:.0f}%]'.format(
                now(), name, g, tries[job], completed_count, n_total, pct), flush=True)
            time.sleep(START_WAIT); used = gpu_used_mb()

        if pending and free and not running and time.time() - last_wait > 600:
            print('[{}] waiting: GPUs busy {}'.format(now(), {g: used.get(g) for g in free}), flush=True)
            last_wait = time.time()
        time.sleep(30)

        for p in list(running):
            if p.poll() is None: continue
            g, job, name, log, t0 = running.pop(p); log.close(); free.append(g)
            mins = (time.time() - t0) / 60
            if p.returncode == 0:
                completed_count += 1
                pct = 100.0 * completed_count / n_total
                # 读验证集最佳分数（如有）
                mp = f'{run_dir(*job)}/metrics.json'
                dev_str = ''
                if os.path.exists(mp):
                    try:
                        _m = json.load(open(mp, encoding='utf-8'))
                        bdev = _m.get('train', {}).get('best_dev_ndcg@10')
                        btest = _m.get('test', {}).get('ndcg@10')
                        if bdev:   dev_str += '  dev={:.2f}'.format(bdev * 100)
                        if btest:  dev_str += '  test={:.2f}'.format(btest * 100)
                    except Exception:
                        pass
                print('[{}] DONE   {:<38s} {:.0f}min{}  [{}/{}  {:.0f}%]'.format(
                    now(), name, mins, dev_str, completed_count, n_total, pct), flush=True)
            elif tries[job] < MAX_TRIES:
                pending.append(job)
                shutil.rmtree(model_dir(*job), ignore_errors=True)
                shutil.rmtree(f'{CKPT_ROOT}/{run_name(*job)}', ignore_errors=True)
                print('[{}] FAIL   {:<38s} exit {} {:.0f}min  retry {}/{}'.format(
                    now(), name, p.returncode, mins, tries[job], MAX_TRIES), flush=True)
            else:
                print('[{}] GIVEUP {:<38s} after {} tries'.format(
                    now(), name, MAX_TRIES), flush=True)

    print('-' * 70, flush=True)
    print('[{}] All done. {}/{} completed.'.format(now(), completed_count, n_total), flush=True)
    report()

# =============================================================================
# 报告
# =============================================================================

def paired_permutation(a, b, iters=PERM_ITERS, seed=0):
    import numpy as np
    d   = np.asarray(a) - np.asarray(b)
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    hits = 0
    for s in range(0, iters, 1000):
        n     = min(1000, iters - s)
        signs = rng.choice([-1., 1.], size=(n, len(d)))
        hits += int((np.abs((signs * d).mean(axis=1)) >= obs).sum())
    return (hits + 1) / (iters + 1)

def paired_bootstrap_ci(a, b, iters=BOOT_ITERS, seed=0):
    import numpy as np
    d   = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    n   = len(d)
    means = []
    for s in range(0, iters, 200):
        m = min(200, iters - s)
        means.append(d[rng.integers(0, n, size=(m, n))].mean(axis=1))
    means = np.concatenate(means)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, mx = [1.] * len(ps), 0.
    for rank, i in enumerate(order):
        mx    = max(mx, min(1., (len(ps) - rank) * ps[i]))
        adj[i] = mx
    return adj

def report():
    import statistics
    os.makedirs(REPORTS, exist_ok=True)
    show = ['ndcg@10', 'mrr@10', 'recall@1', 'recall@10', 'recall@100']
    primary_pairs = {(a, b) for a, b in PRIMARY_COMPARE}
    md  = ['# Step 09 (new) · Results', '',
           'Generated {}. Recipe `{}` (loss {}, batch {}, epochs {}). '
           'Test set: all unique answers as retrieval corpus. Mean ± std over seeds {}.'.format(
               datetime.datetime.now().isoformat(timespec='seconds'),
               RUN_TAG, LOSS_NAME, BATCH_SIZE, EPOCHS, SEEDS), '',
           '> onto_* (all ontology methods): model-independent (shared training file across bge and qwen).', '']
    out = {}

    for m in ALL_MODELS:
        rows, perq, seed_ndcg, by_src_rows, srcs, reused = [], {}, {}, [], [], []
        zs = os.path.join(run_dir(m, 'zeroshot', 0), 'metrics.json')
        if os.path.exists(zs):
            z = load_json(zs)
            rows.append(['zero-shot', '1'] + ['{:.2f}'.format(z['test'][k] * 100) for k in show])

        for meth in ALL_METHODS:
            runs, seeds_ok = [], []
            for s in SEEDS:
                p = os.path.join(run_dir(m, meth, s), 'metrics.json')
                if os.path.exists(p):
                    runs.append(load_json(p)); seeds_ok.append(s)
            if not runs: continue
            cells = []
            for k in show:
                v  = [r['test'][k] * 100 for r in runs]
                sd = statistics.stdev(v) if len(v) > 1 else 0.
                cells.append('{:.2f} ± {:.2f}'.format(statistics.mean(v), sd))
            shared_note = ' †' if meth in SHARED_METHODS else ''
            rows.append([meth + shared_note, str(len(runs))] + cells)
            reused += ['{} seed{}'.format(meth, s) for s, r in zip(seeds_ok, runs) if r.get('reused_from')]
            out.setdefault(m, {})[meth] = {
                **{k: [r['test'][k] for r in runs] for k in METRICS},
                'seeds': seeds_ok, 'model_paths': [r.get('model_path') for r in runs]}
            seed_ndcg[meth] = {s: r['test']['ndcg@10'] for s, r in zip(seeds_ok, runs)}
            acc = defaultdict(list)
            for s in seeds_ok:
                for r in read_jsonl(os.path.join(run_dir(m, meth, s), 'per_query.jsonl')):
                    acc[r['qid']].append(r['ndcg@10'])
            perq[meth] = {q: sum(v) / len(v) for q, v in acc.items()}
            srcs        = sorted(runs[0]['test_by_source'])
            by_src_rows.append([meth] + [
                '{:.2f}'.format(statistics.mean(r['test_by_source'].get(src, {}).get('ndcg@10', 0) for r in runs) * 100)
                for src in srcs])

        if not rows: continue
        md += ['## ' + m, '',
               '| Method | Seeds | ' + ' | '.join(show) + ' |',
               '| :--- | ---: | ' + ' | '.join(['---:'] * len(show)) + ' |']
        md += ['| ' + ' | '.join(r) + ' |' for r in rows] + ['']
        md += ['† model-independent training file (onto_* (all ontology methods) / random / mnrl).', '']
        if reused:
            md += ['Reused from old run: ' + ', '.join(reused) + '.', '']

        comps = []
        for a_, b_ in COMPARE:
            if a_ in perq and b_ in perq:
                qs    = sorted(set(perq[a_]) & set(perq[b_]))
                a     = [perq[a_][q] for q in qs]
                b     = [perq[b_][q] for q in qs]
                delta = (sum(a) - sum(b)) / len(qs) * 100
                lo, hi = paired_bootstrap_ci(a, b)
                p     = paired_permutation(a, b)
                common = sorted(set(seed_ndcg[a_]) & set(seed_ndcg[b_]))
                wins  = sum(seed_ndcg[a_][s] > seed_ndcg[b_][s] for s in common)
                comps.append({'method': a_, 'baseline': b_, 'delta_ndcg@10': delta,
                              'ci95': [lo * 100, hi * 100], 'p': p,
                              'seeds_better': wins, 'seeds_compared': len(common),
                              'noninferior': (lo * 100) > -NONINF_MARGIN,
                              'primary': (a_, b_) in primary_pairs})
        if comps:
            adj = holm([c['p'] for c in comps])
            md += ['Paired comparison on per-query nDCG@10 (averaged over seeds). '
                   'Permutation test {} iterations; 95% paired bootstrap {} resamples; '
                   'Holm correction over {} comparisons. '
                   '★ = pre-declared primary comparison.'.format(PERM_ITERS, BOOT_ITERS, len(comps)), '',
                   '| Method vs baseline | Δ nDCG@10 | 95% CI | p | p (Holm) | Seeds better | Non-inferior (margin -{:.2f}) | Primary |'.format(NONINF_MARGIN),
                   '| :--- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |']
            for c, pa in zip(comps, adj):
                c['p_holm'] = pa
                star = '★' if c['primary'] else ''
                md += ['| {} vs {} | {:+.2f} | [{:+.2f}, {:+.2f}] | {:.4f} | {:.4f} | {}/{} | {} | {} |'.format(
                    c['method'], c['baseline'], c['delta_ndcg@10'],
                    c['ci95'][0], c['ci95'][1], c['p'], pa,
                    c['seeds_better'], c['seeds_compared'],
                    'yes' if c['noninferior'] else 'no', star)]
            md += ['', 'Non-inferior = lower bound of the 95% paired-bootstrap CI of the nDCG@10 difference is above -{:.2f} points. '
                   'The test reflects only the sampling of test questions, not training-seed variation.'.format(NONINF_MARGIN), '']
            out[m]['comparisons'] = comps

        if by_src_rows:
            md += ['nDCG@10 by source (mean over seeds):', '',
                   '| Method | ' + ' | '.join(srcs) + ' |',
                   '| :--- | ' + ' | '.join(['---:'] * len(srcs)) + ' |']
            md += ['| ' + ' | '.join(r) + ' |' for r in by_src_rows] + ['']

    dose_md = []
    for m in ALL_MODELS:
        have = [(meth, pct) for meth, pct in DOSE_LEVELS if meth in out.get(m, {})]
        if len(have) < 2:
            continue
        dose_md += ['## ' + m + ' · nDCG@10 vs injected same-disease negative rate', '',
                    '| Injected same-disease negatives | Method | Seeds | nDCG@10 (mean ± std) |',
                    '| ---: | :--- | ---: | ---: |']
        xs, ys, es = [], [], []
        for meth, pct in have:
            v = [x * 100 for x in out[m][meth]['ndcg@10']]
            mu_ = statistics.mean(v)
            sd_ = statistics.stdev(v) if len(v) > 1 else 0.
            xs.append(pct); ys.append(mu_); es.append(sd_)
            dose_md.append('| {}% | {} | {} | {:.2f} ± {:.2f} |'.format(pct, meth, len(v), mu_, sd_))
        refs = {}
        for rm in ('random', 'mnrl'):
            if rm in out[m]:
                refs[rm] = statistics.mean([x * 100 for x in out[m][rm]['ndcg@10']])
                dose_md.append('| ref | {} | {} | {:.2f} |'.format(rm, len(out[m][rm]['ndcg@10']), refs[rm]))
        dose_md += ['', 'Each sample draws a same-disease (false) negative with the given probability, otherwise a safe-pool negative. 0% = onto_pool_rand. Random sampling from the full corpus gives about 27% same-disease negatives (step 08 report).', '']
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(5.5, 3.8))
            ax.errorbar(xs, ys, yerr=es, marker='o', capsize=3, label='ontology pool + injected same-disease')
            for rm, rv in refs.items():
                ax.axhline(rv, linestyle='--', linewidth=1, label=rm)
            ax.axvline(27, linestyle=':', linewidth=1, color='gray')
            ax.set_xlabel('Injected same-disease negatives (%)')
            ax.set_ylabel('nDCG@10')
            ax.set_title(m)
            ax.legend(fontsize=8)
            fig.tight_layout()
            png = f'{REPORTS}/09_dose_{RUN_TAG}_{m}.png'
            fig.savefig(png, dpi=200)
            plt.close(fig)
            dose_md += ['Curve: `' + png + '`', '']
        except Exception as e:
            dose_md += ['(plot skipped: {})'.format(e), '']
    md += dose_md

    md += ['## Models', '',
           'Saved under `{}/`, named `{}_<model>_<method>_seed<seed>_{}`. '
           'Each contains `training_info.json`.'.format(MODEL_ROOT, DOMAIN, RUN_TAG), '']

    with open(f'{REPORTS}/09_results_{RUN_TAG}.md', 'w', encoding='utf-8') as f:
        f.write(NL.join(md) + NL)
    with open(f'{REPORTS}/09_results_{RUN_TAG}.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(NL.join(md))
    print(NL + 'Report: ' + f'{REPORTS}/09_results_{RUN_TAG}.md')

# =============================================================================

def main():
    if ARGS.launch:
        launch()
    elif ARGS.report:
        report()
    elif ARGS.model and ARGS.method:
        run(ARGS.model, ARGS.method, ARGS.seed, ARGS.max_steps, ARGS.overwrite)
    else:
        print('Use --launch, --report, or --model M --method X --seed S. '
              'In Jupyter call run(model, method, seed) or report().')

if __name__ == '__main__':
    main()
