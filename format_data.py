import os

input_file = '/home/filliones/Downloads/Documents/Work/Implementations/Mamba/data/deu_subset.txt'
output_file = '/home/filliones/Downloads/Documents/Work/Implementations/Mamba/data/formatted_deu.txt'

with open(input_file, 'r', encoding='utf-8') as f:
    lines = f.read().strip().split('\n')

with open(output_file, 'w', encoding='utf-8') as f:
    for line in lines:
        parts = line.split('\t')
        if len(parts) >= 2:
            en, de = parts[0], parts[1]
            f.write(f"[DE] {de} [EN] {en}\n")
