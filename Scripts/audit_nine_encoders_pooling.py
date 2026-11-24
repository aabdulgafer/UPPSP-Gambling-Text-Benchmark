"""Read-only pooling interface audit; run in the original training environment."""
import argparse, csv, gc, inspect, json, platform, traceback
from pathlib import Path
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

MODELS = {'BERT':'bert-base-uncased','RoBERTa':'roberta-base','ALBERT':'albert-base-v1','ELECTRA':'google/electra-small-discriminator','DeBERTa-v3':'microsoft/deberta-v3-base','ERNIE':'nghuyong/ernie-2.0-en','XLNet':'xlnet-base-cased','ModernBERT':'answerdotai/ModernBERT-base','NeoBERT':'chandar-lab/NeoBERT'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='pooling_audit_results')
    p.add_argument('--allow-download', action='store_true')
    p.add_argument('--trust-neobert-code', action='store_true', help='Allow trusted cached NeoBERT custom code, as in original run configuration')
    p.add_argument('--runs-dir', type=Path, default=Path('runs'))
    a = p.parse_args()
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    configs = []
    for f in a.runs_dir.rglob('run_config.json'):
        try:
            c = json.loads(f.read_text(encoding='utf-8-sig'))
            if c.get('model_name') in MODELS.values():
                configs.append({'path':str(f), **{k:c.get(k) for k in ('model_name','seed','pooling_strategy','model_type','max_length','dropout','trust_remote_code','model_revision','environment','runner_sha256')}})
        except Exception as e:
            print(f'Config read warning: {f}: {e}')
    (out/'training_config_evidence.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
    rows=[]; details={}
    for name, mid in MODELS.items():
        print(f'Auditing {name}: {mid}',flush=True)
        row={'encoder':name,'model_id':mid,'status':'FAILED'}
        model=tok=output=batch=hidden=pooled=native=None
        try:
            matching=[c for c in configs if c['model_name']==mid and c['pooling_strategy'] in ('last_token','pooler_output_or_first_token')]
            revisions={c['model_revision'] for c in matching}
            if len(revisions)>1: raise ValueError(f'Multiple recorded revisions: {revisions}')
            revision=next(iter(revisions),None)
            kw={'local_files_only':not a.allow_download}
            if name in ('ModernBERT','NeoBERT'):
                kw.update(revision=revision,trust_remote_code=(name=='NeoBERT' and a.trust_neobert_code))
            tok=AutoTokenizer.from_pretrained(mid,**kw)
            model=AutoModel.from_pretrained(mid,**kw).cpu().eval()
            batch=tok(['I lost again today.','I decided to stop playing after losing money.'],padding='max_length',truncation=True,max_length=64,return_tensors='pt')
            batch={k:v for k,v in batch.items() if k in ('input_ids','attention_mask','token_type_ids')}
            with torch.inference_mode():
                output=model(**batch)
                hidden=output.last_hidden_state
                native=getattr(output,'pooler_output',None)
                if model.config.model_type=='xlnet':
                    strategy='final_position'; pooled=hidden[:,-1]
                    assert tok.cls_token_id is not None and bool((batch['input_ids'][:,-1]==tok.cls_token_id).all()), 'XLNet CLS is not at final position'
                elif native is not None:
                    strategy='native_pooler_output'; pooled=native
                else:
                    strategy='first_position'; pooled=hidden[:,0]
                pooler=getattr(model,'pooler',None)
                modules={n:repr(m) for n,m in model.named_modules() if 'pooler' in n.lower() and '.' not in n}
                sources={}
                for n,m in [('model_forward',model),('pooler_forward',pooler)]:
                    if m is not None:
                        try: sources[n]=inspect.getsource(type(m).forward)
                        except Exception as e: sources[n]=str(e)
                row.update(status='OK',model_class=type(model).__name__,model_type=model.config.model_type,tokenizer_class=type(tok).__name__,padding_side=tok.padding_side,cls_token_id=tok.cls_token_id,first_token_ids=batch['input_ids'][:,0].tolist(),last_token_ids=batch['input_ids'][:,-1].tolist(),first_attention_mask=batch['attention_mask'][:,0].tolist(),last_attention_mask=batch['attention_mask'][:,-1].tolist(),output_keys=list(output.keys()),native_pooler_available=native is not None,selected_strategy=strategy,hidden_shape=list(hidden.shape),selected_shape=list(pooled.shape),pooler_module=repr(pooler),resolved_commit=getattr(model.config,'_commit_hash',None),requested_revision=revision,matching_training_configs=len(matching))
                if native is not None and pooler is not None:
                    dense=getattr(pooler,'dense',None)
                    if isinstance(dense,torch.nn.Linear):
                        candidate=torch.tanh(dense(hidden[:,0]))
                        row['dense_tanh_first_token_matches']=bool(torch.allclose(candidate,native,atol=1e-5,rtol=1e-4))
                details[name]={'summary':row,'pooler_modules':modules,'source':sources,'config':model.config.to_dict()}
        except Exception as e:
            row.update(status='FAILED',error=repr(e)); details[name]={'summary':row,'traceback':traceback.format_exc()}; print(f'FAILED: {e}',flush=True)
        finally:
            rows.append(row)
            model=tok=output=batch=hidden=pooled=native=None
            gc.collect()
        (out/'pooling_details.json').write_text(json.dumps(details,indent=2,default=str),encoding='utf-8')
        with (out/'pooling_summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r))); w.writeheader(); w.writerows(rows)
    (out/'audit_metadata.json').write_text(json.dumps({'python':platform.python_version(),'platform':platform.platform(),'torch':torch.__version__,'transformers':transformers.__version__,'device':'cpu','max_length':64,'local_files_only':not a.allow_download,'successful':sum(r['status']=='OK' for r in rows),'models':MODELS},indent=2),encoding='utf-8')
    print(f"Completed {sum(r['status']=='OK' for r in rows)}/9. Results: {out.resolve()}")
    return int(any(r['status']!='OK' for r in rows))

if __name__=='__main__':
    raise SystemExit(main())
