---
language: en
---

# PubTator3 dataset

[PubTator3](https://ftp.ncbi.nlm.nih.gov/pub/lu/PubTator3/) annotations.

This dataset is comprised of 10 splits, one for each BioCXML archive file hosted by PubTator3, named "BioCXML_{n}" for `n in list(range(10))`.
The dataset contains titles, abstracts, publication year, and annotation data for each annotation predicted by PubTator3.
If an abstract was split into multiple parts in the PubTator3 archive files, they have been joined so each publication has exactly one abstract.
In addition to PubTator3 data, this has been enriched with reference data pulled from the [PubMed](https://pubmed.ncbi.nlm.nih.gov/) XML files.

## Annotations

Annotations are stored in two parts, the labels and metadata.
Labels are a `Sequence` of `ClassLabel`s.
For each publication, there will be a list of integer IDs for each label type.
To get the normal string identifier, use `dataset[{split}].features[{annotation_type}].feature.int2str`.
For example if "gene" is the annotation type, this will return a NCBI gene ID.
Label keys are named are named after the annotation name as provided by PubTator3 but lower cased.
This does not contain `snp`, `proteinmutation`, or `dnamutatation` annotations because these annotations' identifiers tended to be chained together (multiple identifiers separated by a ";") making it difficult to work with.

In addition to the separate labels for each annotation type, there is a single "annotation" key.
Each publication gets a list of annotation metadata dictionaries.
The dictionaries contain "offset", "length", and "type".
Offsets and length are where in the abstract the annotation was found by PubTator3.

## Masking

As an example of using this data, text can be masked to redact labels for training purposes.

The following will replace all "gene" and "cellline" annotations with the mask token used by BERT tokenizers, "[MASK]"

``` python
def mask_example(
    example: dict[str, list[Any]],
    ann_types: Sequence[str],
    mask_token: str,
):
    abstract = example["abstract"]
    mask = np.ones((len(abstract),))
    for ann in example["annotation"]:
        if ann["type"].lower() not in ann_types:
            continue

        pos_start = ann["offset"]
        pos_end = pos_start + ann["length"]
        mask[pos_start + 1 : pos_end] = 0
        mask[pos_start] = -1

    return {
        "abstract": "".join(
            (
                letter if mask[i] > 0 else mask_token
                for i, letter in enumerate(abstract)
                if mask[i]
            )
        )
    }


mask_token = "[MASK]"
ann_types = ["gene", "cellline"]

dataset = dataset.map(
    mask_examples,
    batched=False,
    num_proc=max_cpu,
    fn_kwargs={"ann_types": ann_types, "mask_token": mask_token},
)
```

## See also

For other examples, see the `dataset.mutators` module of my [abstract2gene](https://github.com/net-synergy/abstract2gene) package.

Also see the `dataset._bioc` module and `example/create_from_bioc` files in the abstract2gene package for the source code that generated this dataset.

For collecting PubMed data, see my [pubmedparser](https://github.com/net-synergy/pubmedparser) package
