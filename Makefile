PYTHON=python

.PHONY: all
all: experiments

.PHONY: dataset
dataset:
	$(PYTHON) example/create_from_bioc.py

.PHONY: train
train:
	$(PYTHON) example/training/embedding_model_selection.py
	$(PYTHON) example/training/finetune_experiments.py \
		--models MPNet PubMedNCL
	$(PYTHON) example/training/finetune_experiments.py \
		--n_steps 20_000 \
		--experiments 5
	$(PYTHON) example/training/plot_training_curve.py
	$(PYTHON) example/training/label_embedding_similarity.py
	$(PYTHON) example/training/train_abstract2gene.py

.PHONY: upload
upload:
	$(PYTHON) example/upload_to_hub.py

.PHONY: experiments
experiments:
	$(PYTHON) example/experiments/test_abstract2gene.py
	$(PYTHON) example/experiments/differential_expression.py
	$(PYTHON) example/experiments/predict_genes_in_behavioral_studies.py
	$(PYTHON) example/analyze_citation_network.py

.PHONY: clean
clean:
	rm -rf models
	rm -rf logs/_tmp

.PHONY: clean-dist
clean-dist: clean
	rm -rf logs
	rm -rf figures
	rm -rf results
	rm -rf dist
