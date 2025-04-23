PYTHON=python

.PHONY: all
all: experiments

.PHONY: dataset
dataset:
	$(PYTHON) example/create_from_bioc.py

.PHONY: train
train:
	$(PYTHON) example/training/embedding_model_selection.py --save false
	# Further test the best performing models in initial test
	$(PYTHON) example/training/embedding_model_selection.py \
		--n_steps 1000 \
		--n_trials 30 \
		--models PubMedNCL MPNet
	$(PYTHON) example/training/finetune_experiments.py
	$(PYTHON) example/training/plot_training_curve.py
	# Further train the best performing experiment
	# (doesn't converge by 10_000 steps).
	$(PYTHON) example/training/finetune_experiments.py \
		--n_steps 20_000 \
		--models MPNet \
		--experiments 3
	$(PYTHON) example/training/label_embedding_similarity.py
	$(PYTHON) example/training/train_abstract2gene.py

.PHONY: upload
upload:
	$(PYTHON) example/upload_to_hub.py

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
