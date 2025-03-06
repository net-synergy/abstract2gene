"""Script to run all analyses.

To ensure reproducible results, each script is passed a random seed to use. The
random seed is a function of the script run order (the ith script gets 10 * i
as a seed). By using multiples of 10 each script can use up to 10 seeds if it
needs more than one.
"""

import subprocess
import sys


def run_script(name: str, i: int):
    command = [sys.executable, name, str(i * 10)]
    outfile = f"{name.split('.')[0]}_out.txt"
    with open(outfile, "w") as results:
        subprocess.run(command, stdout=results, text=True)


scripts = [
    "create_from_bioc.py",
    "embedding_model_selection.py",
    "finetune_encoder.py",
    "train_abstract2gene.py",
    "test_abstract2gene.py",
    "label_embedding_similarity.py",
    "reference_similarity.py",
    "predict_genes_in_behavioral_studies.py",
    "differential_expression.py",
]

for i, script in enumerate(scripts):
    run_script(script, i)
