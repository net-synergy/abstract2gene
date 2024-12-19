import abstract2gene as a2g

DATASET = "bioc"

# Guarantees there is at least 32 publications left to create a template from
# for each gene when training.
MAX_GENE_TESTS = 100

data = a2g.dataset.load_dataset(
    DATASET,
    seed=42,
    batch_size=64,
    template_size=32,
)

model: a2g.model.Model = a2g.model.ModelNoWeights(name="noweights")
a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS, save_results=False)

model = a2g.model.ModelSingleLayer(name="", seed=12, n_dims=20)

# Coarse dimension search
for d in range(1, 20, 2):
    data.reset_rng()
    model.name = f"random_weights_{d}_dims"
    a2g.model.train(model, data, learning_rate=1e-4, max_epochs=100)
    a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS)

# Finer dimension search based on coarse results
for d in range(20, 110, 10):
    data.reset_rng()
    model.name = f"random_weights_{d}_dims"
    a2g.model.train(model, data, learning_rate=1e-4, max_epochs=100)
    a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS)


dims = (data.n_features, 64, 64)
model = a2g.model.ModelMultiLayer(name="multi", seed=20, dims=dims)
a2g.model.train(model, data, learning_rate=1e-4, max_epochs=100)
a2g.model.test(model, data, max_num_tests=MAX_GENE_TESTS)
