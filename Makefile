PYTHON=python

.PHONY: all
all: experiments

.PHONY: dataset
dataset:
	$(PYTHON) example/create_from_bioc.py

.PHONY: train
train:
	$(PYTHON) example/training/embedding_model_selection.py
	$(PYTHON) example/training/finetune_encoder.py
	$(PYTHON) example/training/plot_training_curve.py
	$(PYTHON) example/training/label_embedding_similarity.py
	$(PYTHON) example/training/train_abstract2gene.py

.PHONY: experiments
experiments:
	$(PYTHON) example/experiments/test_abstract2gene.py
	$(PYTHON) example/experiments/reference_similarity.py
	$(PYTHON) example/experiments/differential_expression.py
	$(PYTHON) example/experiments/predict_genes_in_behavioral_studies.py

.PHONY: clean
clean:
	rm -rf models
	rm -rf logs/_tmp

.PHONY: clean-dist
clean-dist: clean
	rm -rf logs
	rm -rf figures
	rm -rf results
