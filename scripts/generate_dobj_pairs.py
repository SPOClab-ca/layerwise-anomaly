import checklist
from checklist.editor import Editor

VERBS = ['gave', 'sent', 'mailed', 'took', 'brought', 'showed', 'sold']

editor = Editor()

for vb in VERBS:
  ret = editor.suggest(
    f'The {{mask}} {vb} the {{mask}} a {{mask}}. The {{mask}} {vb} a {{mask}} to the {{mask}}.',
    nsamples=1000 # Todo: this parameter seems to be broken with suggest() but works for template()
  )
  for t in ret:
    if t[0] != t[3]: continue
    if t[1] != t[5]: continue
    if t[2] != t[4]: continue
    if t[0] == t[1] or t[0] == t[2] or t[1] == t[2]: continue
    subj = t[0]
    iobj = t[1]
    dobj = t[2]
    print(f'The {subj} {vb} the {iobj} a {dobj}. The {subj} {vb} a {dobj} to the {iobj}.',)
