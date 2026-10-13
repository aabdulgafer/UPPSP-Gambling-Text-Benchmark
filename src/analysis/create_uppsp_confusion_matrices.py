"""Reproduce five UPPS-P user-level confusion matrices from frozen ModernBERT results.

Uses original pooling, seeds 13/42/87, and the frozen 300-user test set.
Predictions are obtained by averaging per-monologue class probabilities within user,
then argmax. The per-seed confusion matrices are summed; display is row-normalized.
The sum is 900 user-seed evaluations, NOT 900 independent users.
"""
from pathlib import Path
from zipfile import ZipFile
from io import BytesIO
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/mnt/data')
ARCHIVE = ROOT / 'contemporary_extension_final_test_results_v1.zip'
TRAITS = [('negative_urgency', 'Negative urgency'), ('positive_urgency','Positive urgency'), ('lack_of_premeditation','Lack of premeditation'), ('lack_of_perseverance','Lack of perseverance'), ('sensation_seeking','Sensation seeking')]
LEVELS = ['low','mid','high']
SEEDS = [13,42,87]
conf = {t:np.zeros((3,3),dtype=int) for t,_ in TRAITS}
checks=[]
with ZipFile(ARCHIVE) as z:
    for seed in SEEDS:
        root = f'contemporary_extension_final_test_results_v1/modernbert_seed{seed}/'
        df = pd.read_csv(BytesIO(z.read(root+'test_predictions.csv')))
        metrics = json.loads(z.read(root+'test_metrics.json'))
        assert df.user_id.nunique()==300 and len(df)==9441
        macro=[]
        for t,_ in TRAITS:
            g=df.groupby('user_id',sort=True)
            y=g[t+'_true'].first().to_numpy()
            probs=g[[f'{t}_prob_{level}' for level in LEVELS]].mean().to_numpy()
            pred=np.asarray(LEVELS)[probs.argmax(axis=1)]
            macro.append(f1_score(y,pred,labels=LEVELS,average='macro'))
            conf[t] += confusion_matrix(y,pred,labels=LEVELS)
        actual=np.mean(macro)
        expected=metrics['user_level']['overall_macro_f1']
        assert abs(actual-expected)<1e-10,(seed,actual,expected)
        checks.append({'seed':seed,'users':300,'monologues':9441,'recomputed_macro_f1':actual,'saved_macro_f1':expected})

fig,axs=plt.subplots(2,3,figsize=(13.4,9.4),dpi=160)
fig.subplots_adjust(left=.075,right=.98,top=.89,bottom=.15,wspace=.38,hspace=.48)
fig.patch.set_facecolor('white')
for i,(t,title) in enumerate(TRAITS):
    ax=axs.flat[i]
    cm=conf[t]
    row=cm.sum(axis=1,keepdims=True)
    pct=cm/row*100
    ax.imshow(pct,vmin=0,vmax=100,cmap='Blues',aspect='equal')
    ax.set_title(title,fontsize=12.5,fontweight='semibold',pad=11)
    ax.set_xticks(range(3),['Low','Mid','High'])
    ax.set_yticks(range(3),['Low','Mid','High'])
    ax.set_xlabel('Predicted level',fontsize=10)
    ax.set_ylabel('True level',fontsize=10)
    ax.tick_params(length=0,labelsize=10)
    for r in range(3):
        for c in range(3):
            val=pct[r,c]
            ax.text(c,r,f'{val:.1f}%\n({cm[r,c]})',ha='center',va='center',fontsize=10,color='white' if val>=52 else '#142c42',fontweight='semibold' if r==c else 'normal')
    ax.set_xticks(np.arange(-.5,3,1),minor=True)
    ax.set_yticks(np.arange(-.5,3,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=2)
    ax.tick_params(which='minor',bottom=False,left=False)
    for s in ax.spines.values():s.set_visible(False)
axs.flat[-1].axis('off')
fig.suptitle('UPPS-P facet-level confusion matrices',fontsize=18,fontweight='bold')
fig.text(.5,.055,'ModernBERT · original pooling · 300 held-out users · 3 seeds (13, 42, 87)\nRows show recall (%) within each true level; cell counts sum across seeds (900 user-seed evaluations).',ha='center',fontsize=10,color='#334155')
fig.savefig(ROOT/'uppsp_modernbert_confusion_matrices.png',dpi=240,bbox_inches='tight',facecolor='white')
fig.savefig(ROOT/'uppsp_modernbert_confusion_matrices.pdf',bbox_inches='tight',facecolor='white')

rows=[]
for t,title in TRAITS:
    cm=conf[t]
    for i,tru in enumerate(LEVELS):
        for j,pred in enumerate(LEVELS):
            rows.append({'facet':title,'true_level':tru,'predicted_level':pred,'count_across_3_seeds':int(cm[i,j]),'row_percentage':round(100*cm[i,j]/cm[i].sum(),4)})
pd.DataFrame(rows).to_csv(ROOT/'uppsp_modernbert_confusion_matrices_data.csv',index=False)
print(pd.DataFrame(checks).to_string(index=False))
for t,title in TRAITS:
    cm=conf[t]
    print(title, 'accuracy',round(np.trace(cm)/cm.sum(),4),'per-level recalls',np.round(np.diag(cm)/cm.sum(axis=1),3),'support',cm.sum(axis=1).tolist())
print('Created PNG, PDF, CSV')
