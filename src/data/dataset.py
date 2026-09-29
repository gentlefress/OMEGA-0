"""Dataset composition used by the training loader."""
from __future__ import annotations
from torch.utils.data import Dataset as TorchDataset

class TransformedDataset(TorchDataset):
    def __init__(self, dataset, transforms, kwargs):
        self.dataset, self.transforms, self.kwargs = dataset, transforms, kwargs

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        value = self.dataset[index]
        for transform in self.transforms:
            value = transform(value, **self.kwargs)
        return value


class MixtureDataset(TorchDataset):

    def __init__(self, datasets: dict[TorchDataset, float], num_samples_per_epoch: int) -> None:
        self.datasets = list(datasets.keys())
        self.ratios = list(datasets.values())
        self.num_samples_per_epoch = num_samples_per_epoch

    def __getitem__(self, index):
        assert isinstance(index, tuple) and len(index) == 2, 'Maybe forget to use batch sampler? see ../data/sampler.py'
        (dataset_id, sample_id) = index
        return self.datasets[dataset_id][sample_id]

    def __len__(self):
        return self.num_samples_per_epoch


def dataset(reader, transforms, processor=None, *, vlm_processor=None, action_tokenizer=None):
    from omega.models.external import resolve_factory

    source = resolve_factory(reader["factory"])(**reader.get("options", {}))
    pipeline = [resolve_factory(spec["factory"])(**spec.get("options", {})) for spec in transforms]
    kwargs = {}
    if processor:
        if vlm_processor is not None:
            raise ValueError("Pretraining supplies its model's processor; remove data.options.processor")
        from transformers import AutoProcessor
        vlm_processor = AutoProcessor.from_pretrained(**processor)
    if vlm_processor is not None:
        kwargs["vlm_processor"] = vlm_processor
    if action_tokenizer is not None:
        kwargs["action_tokenizer"] = action_tokenizer
    return TransformedDataset(source, pipeline, kwargs)
