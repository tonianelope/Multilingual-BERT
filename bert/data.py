import codecs
from pathlib import Path

import pandas as pd


# from https://github.com/sberbank-ai/ner-bert/blob/master/examples/conll-2003.ipynb
def read_conll_data(input_file:str):
    """Reads CONLL-2003 format data."""
    with codecs.open(input_file, "r", encoding="utf-8") as f:
        lines = []
        words = []
        labels = []
        for line in f:
            contends = line.strip()
            word = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            if contends.startswith("-DOCSTART-"):
                words.append('')
                continue

            if len(contends) == 0 and not len(words):
                words.append("")

            if len(contends) == 0 and words[-1] == '.':
                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue
            words.append(word)
            labels.append(label.replace("-", "_"))
        return lines

def conll_to_csv(csv_dir, file_names:list):
    """Write CONLL-2003 to csv"""

    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)

    for file in file_names:
        file = Path(file)
        if(file.is_file()):
            data = read_conll_data(file)
            df = pd.DataFrame(data, columns=['labels', 'text'])

            # csv_path = csv_dir / (file.name + '.csv')
            df.to_csv(csv_path, index=False)
            print(f'Wrote {csv_path}')
        else:
            raise ValueError(f'{file} does not exist, or is not a file')
