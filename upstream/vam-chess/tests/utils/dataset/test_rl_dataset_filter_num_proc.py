# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class _DummyDataset:
    def __init__(self):
        self.filter_kwargs = None

    def filter(self, fn, **kwargs):
        # The function itself is irrelevant for this unit test; we only care about
        # whether `num_proc` is passed (which triggers multiprocessing Pool creation).
        self.filter_kwargs = dict(kwargs)
        return self

    def __len__(self):
        return 0


class _DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        # Never executed by this test (we don't evaluate the filter lambda), but
        # keep a plausible signature to avoid accidental AttributeError if this
        # changes in the future.
        return [0]


def _make_dataset_with_workers(num_workers: int):
    from verl.utils.dataset.rl_dataset import RLHFDataset

    ds = RLHFDataset.__new__(RLHFDataset)
    ds.filter_overlong_prompts = True
    ds.tokenizer = _DummyTokenizer()
    ds.processor = None
    ds.prompt_key = "prompt"
    ds.image_key = "images"
    ds.video_key = "videos"
    ds.apply_chat_template_kwargs = {}
    ds.tool_schemas = None
    ds.num_workers = num_workers
    ds.max_prompt_length = 32
    return ds


def test_rl_dataset_filter_does_not_pass_num_proc_when_workers_is_1():
    ds = _make_dataset_with_workers(num_workers=1)
    dummy = _DummyDataset()
    ds.maybe_filter_out_long_prompts(dummy)
    assert dummy.filter_kwargs is not None
    assert "num_proc" not in dummy.filter_kwargs
    assert "desc" in dummy.filter_kwargs


def test_rl_dataset_filter_passes_num_proc_when_workers_greater_than_1():
    ds = _make_dataset_with_workers(num_workers=2)
    dummy = _DummyDataset()
    ds.maybe_filter_out_long_prompts(dummy)
    assert dummy.filter_kwargs is not None
    assert dummy.filter_kwargs.get("num_proc") == 2
    assert "desc" in dummy.filter_kwargs

