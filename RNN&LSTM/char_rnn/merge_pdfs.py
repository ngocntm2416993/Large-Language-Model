import glob, re, unicodedata
from pypdf import PdfReader
parts = []
for pdf in sorted(glob.glob('*.pdf')):
    t = '\n'.join(p.extract_text() or '' for p in PdfReader(pdf).pages)
    t = unicodedata.normalize('NFC', t)
    t = re.sub(r'^\s*\d+\s*$', '', t, flags=re.M)
    t = re.sub(r'-\n(\w)', r'\1', t)
    t = re.sub(r'[^\S\n]+', ' ', t)
    t = re.sub(r' *\n *', '\n', t)
    t = re.sub(r'\n{3,}', '\n\n', t)
    t = re.sub(r'(?<!\n)\n(?!\n|[-–—•])', ' ', t)
    print(pdf, len(t), 'ký tự')
    parts.append(t.strip())
all_t = '\n\n'.join(parts)
open('all_books.txt', 'w', encoding='utf-8').write(all_t)
print('TỔNG:', len(all_t), 'ký tự | Tab:', all_t.count('\t'))