import checklist
from checklist.editor import Editor

editor = Editor()

ret = editor.template('The {mask} gave the {mask} a {mask}.')
for sent in ret.data[:10]:
  print(sent)

print()

ret = editor.template('The {mask} gave a {mask} to the {mask}.')
for sent in ret.data[:10]:
  print(sent)
