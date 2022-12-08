# coding=utf-8
# Copyright 2020 BigScience Contributors.
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
"""P3 (Public Pool of Prompts)"""

import os
import datasets
import json
import urllib
from collections import defaultdict

_CITATION = """@misc{sanh2021multitask,
      title={Multitask Prompted Training Enables Zero-Shot Task Generalization},
      author={Victor Sanh and Albert Webson and Colin Raffel and Stephen H. Bach and Lintang Sutawika and Zaid Alyafeai and Antoine Chaffin and Arnaud Stiegler and Teven Le Scao and Arun Raja and Manan Dey and M Saiful Bari and Canwen Xu and Urmish Thakker and Shanya Sharma Sharma and Eliza Szczechla and Taewoon Kim and Gunjan Chhablani and Nihal Nayak and Debajyoti Datta and Jonathan Chang and Mike Tian-Jian Jiang and Han Wang and Matteo Manica and Sheng Shen and Zheng Xin Yong and Harshit Pandey and Rachel Bawden and Thomas Wang and Trishala Neeraj and Jos Rozen and Abheesht Sharma and Andrea Santilli and Thibault Fevry and Jason Alan Fries and Ryan Teehan and Stella Biderman and Leo Gao and Tali Bers and Thomas Wolf and Alexander M. Rush},
      year={2021},
      eprint={2110.08207},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}"""

_DESCRIPTION = """\
P3 (Public Pool of Prompts) is a collection of prompted English datasets covering a diverse set of NLP tasks. A prompt is the combination of an input template and a target template. The templates are functions mapping a data example into natural language for the input and target sequences. For example, in the case of an NLI dataset, the data example would include fields for *Premise, Hypothesis, Label*. An input template would be *If {Premise} is true, is it also true that {Hypothesis}?*, whereas a target template can be defined with the label choices *Choices[label]*. Here *Choices* is prompt-specific metadata that consists of the options *yes, maybe, no* corresponding to *label* being entailment (0), neutral (1) or contradiction (2).

Prompts are collected using [Promptsource](https://github.com/bigscience-workshop/promptsource), an interface to interactively write prompts on datasets, and collect prompt-specific metadata such as evaluation metrics. As of October 13th, there are 2'000 prompts collected for 270+ data(sub)sets. The collection of prompts of P3 is publicly available on [Promptsource](https://github.com/bigscience-workshop/promptsource).

To train [T0*](https://huggingface.co/bigscience/T0pp), we used a subset of the prompts available in Promptsource (see details [here](https://huggingface.co/bigscience/T0pp#training-data)). However, some of the prompts use `random.choice`, a method that selects uniformly at random an option in a list of valid possibilities. For reproducibility purposes, we release the collection of prompted examples used to train T0*. **The data available here are the materialized version of the prompted datasets used in [Multitask Prompted Training Enables Zero-Shot Task Generalization](https://arxiv.org/abs/2110.08207) which represent only a subset of the datasets for which there is at least one prompt in Promptsource.**
"""

_LICENSE = "Apache License 2.0"

_HOMEPAGE = "https://github.com/bigscience-workshop/promptsource"

_MAX_DATASET_SIZE=500000
_RAW_DATA_PATH = "/sharefs/english/yanan/data/P3"
_DATA_PATH = "/sharefs/english/yanan/data/P3/data"
# _HUB_PATH = "https://huggingface.co/datasets/bigscience/P3/raw/main"


logger = datasets.logging.get_logger(__name__)


def load_cached_task(features_dict, tfrecord):
    import tensorflow as tf
    # Use `FixedLenSequenceFeature` for sequences with variable length.
    def _feature_config(shape, dtype):
        if dtype in ("int32", "bool"):
            # int32 and bool are stored as int64 in the tf.train.Example protobuf.
            dtype = "int64"
        if shape and shape[0] is None:
            return tf.io.FixedLenSequenceFeature(
                shape[1:], dtype, allow_missing=True
            )
        return tf.io.FixedLenFeature(shape, dtype)

    feature_description = {
        feat: _feature_config(**desc) for feat, desc in features_dict.items()
    }

    ds = tf.data.TFRecordDataset(tf.io.gfile.glob([tfrecord])) # TODO -> handle multiple shards
    ds = ds.map(
        lambda pb: tf.io.parse_single_example(pb, feature_description),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # Cast features back to the types from the info JSON since some features
    # must be cast for storage (e.g., int32 is stored as int64).
    ds = ds.map(
        lambda x: {k: tf.cast(v, features_dict[k]["dtype"]) for k, v in x.items()},
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    return ds

"""
def read_from_url(url):
    try:
        content = urllib.request.urlopen(url, timeout=10.0)
        logger.info(f"Downloaded {url}")
    except urllib.error.URLError as e:
        raise ConnectionError(e)
    return content.read().decode("utf-8")
"""

def find_task_splits_and_features_dict():
    """Get the task available (list was pre-computed by `print_data_split_sizes.py`), and get the features for each task."""
    task_splits_and_features = defaultdict(dict)
    targets_max_tokens_dict = defaultdict(dict)
    file_path = os.path.join(_RAW_DATA_PATH, "data_split_sizes.csv")
    task_to_split_dict = {}
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            if idx == 0:
                continue
            line = line.strip()
            line_splits = line.split("|")
            task_to_split_dict[line_splits[0]] = json.loads(line_splits[1])

    for task_name, split_sizes in task_to_split_dict.items():
        for split_name in split_sizes.keys():
            ## TODO: change the path
            split_file_path=f"{_DATA_PATH}/{task_name}/info.{split_name}.json"
            split_info = json.loads(open(split_file_path, "r").read())
            features_dict = split_info["features"]
            assert split_info["num_shards"] == 1 # TODO -> handle multiple shards
            if not task_splits_and_features[task_name]:
                task_splits_and_features[task_name] = {
                    "splits": [],
                    "features_dict": features_dict,
                }
            task_splits_and_features[task_name]["splits"].append(split_name)
            assert features_dict == task_splits_and_features[task_name]["features_dict"]
        info_file_path=f"{_DATA_PATH}/{task_name}/stats.train.json"
        split_info = json.loads(open(info_file_path, "r").read())
        targets_max_tokens = split_info["targets_max_tokens"]
        targets_max_tokens_dict[task_name] = int(targets_max_tokens)
    return task_splits_and_features, targets_max_tokens_dict


_TASK_SPLITS_AND_FEATURES_DICT, _TASK_TARGET_MAX_TOKENS_DICT = find_task_splits_and_features_dict()
P3_TASK_LIST = list(_TASK_SPLITS_AND_FEATURES_DICT.keys())
_URLs = {
    task_name: {
        split_name: {
            "tfrecord": f"{_DATA_PATH}/{task_name}/{split_name}.tfrecord-00000-of-00001", # TODO -> handle multiple shards
        }
        for split_name in splits_and_features_dict["splits"]
    }
    for task_name, splits_and_features_dict in _TASK_SPLITS_AND_FEATURES_DICT.items()
}

datasets_without_validation = [
    "ag_news", "dbpedia_14", "trec", "amazon_polarity", "imdb", "yelp_review_full", "wiki_bio",
    "web_questions"]

large_t0_tasks = ["gigaword", "amazon_polarity", "wiki_bio", "dbpedia_14", "yelp_review_full"]
large_t0_tasks_prompt_count = {}
for large_task in large_t0_tasks:
    for task_name in P3_TASK_LIST:
        if task_name.startswith(large_task):
            if large_task not in large_t0_tasks_prompt_count:
                large_t0_tasks_prompt_count[large_task] = 1
            else:
                large_t0_tasks_prompt_count[large_task] += 1

large_t0_task_dict = {}
for cur_task in P3_TASK_LIST:
    for large_task_prefix in large_t0_tasks_prompt_count.keys():
        if cur_task.startswith(large_task_prefix):
            large_t0_task_dict[cur_task] = int(_MAX_DATASET_SIZE / large_t0_tasks_prompt_count[large_task_prefix])

DEBUG_TRAIN_TASK_NAME = ["ropes"]
DEBUG_TRAIN_TASK_LIST=[]
for task_name in DEBUG_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    DEBUG_TRAIN_TASK_LIST = DEBUG_TRAIN_TASK_LIST + sub_list


T0_TRAIN_TASK_NAME = [
    "ag_news",
    "app_reviews",
    "wiki_bio",
    "cnn_dailymail/3.0.0",
    "gigaword",
    "wiki_hop/original",
    "glue/mrpc",
    "glue/qqp",
    "amazon_polarity",
    "paws/labeled_final",
    "dbpedia_14",
    "dream",
    "kilt_tasks/hotpotqa",
    "trec",
    "multi_news",
    "samsum",
    "xsum",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "wiki_qa",
    "common_gen",
    "adversarial_qa/dbidaf",
    "adversarial_qa/dbert",
    "adversarial_qa/droberta",
    "quoref",
    "ropes",
    "duorc/SelfRC",
    "duorc/ParaphraseRC",
    "sciq",
    "quarel",
    "qasc",
    "cosmos_qa",
    "wiqa",
    "social_i_qa",
    "quail",
    "quartz",
    # "tydiqa",
    "cos_e/v1.11"
]
T0_TRAIN_TASK_LIST=[]
for task_name in T0_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_TRAIN_TASK_LIST = T0_TRAIN_TASK_LIST + sub_list

BIG_BENCH_TEST_TASK_LIST = [
    "code_line_description",
    "conceptual_combinations",
    "hindu_knowledge",
    "known_unknowns",
    "language_identification",
    "logic_grid_puzzle",
    "logical_deduction",
    "misconceptions",
    "movie_dialog_same_or_different",
    "novel_concepts",
    "strategyqa",
    "formal_fallacies_syllogisms_negation",
    "vitaminc_fact_verification",
    "winowhy"
]

T0_TEST_TASK_NAME = [
    "super_glue/wsc.fixed",
    "super_glue/wic",
    "super_glue/copa",
    # "story_cloze/2016",
    "super_glue/cb",
    "super_glue/rte",
    "hellaswag",
    "anli_r1",
    "anli_r2",
    "anli_r3",
    "winogrande/winogrande_xl",
]

T0_TEST_TASK_LIST=[]
for task_name in T0_TEST_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_TEST_TASK_LIST = T0_TEST_TASK_LIST + sub_list

T0_PLUS_TRAIN_TASK_NAME = [
    "glue/mrpc",
    "glue/qqp",
    "paws/labeled_final",
    "ai2_arc/ARC_Challenge",
    "ai2_arc/ARC_Easy",
    "kilt_tasks/hotpotqa",
    "trivia_qa/unfiltered",
    "web_questions",
    "wiki_qa",
    "adversarial_qa/dbidaf",
    "adversarial_qa/dbert",
    "adversarial_qa/droberta",
    "duorc/SelfRC",
    "duorc/ParaphraseRC",
    "ropes",
    "squad_v2",
    "quoref",
    # "tydiqa",
    "cos_e/v1.11",
    "cosmos_qa",
    "dream",
    "openbookqa/main",
    "qasc",
    "quail",
    "quarel",
    "quartz",
    "race/high",
    "race/middle",
    "sciq",
    "social_i_qa",
    "wiki_hop/original",
    "wiqa",
    "piqa",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "hellaswag",
    "common_gen",
    "wiki_bio",
    "cnn_dailymail/3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "ag_news",
    "dbpedia_14",
    "trec"
]

T0_PLUS_TRAIN_TASK_LIST=[]
for task_name in T0_PLUS_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_PLUS_TRAIN_TASK_LIST = T0_PLUS_TRAIN_TASK_LIST + sub_list

T0_PLUS_PLUS_TRAIN_TASK_NAME = [
    "glue/mrpc",
    "glue/qqp",
    "paws/labeled_final",
    "ai2_arc/ARC_Challenge",
    "ai2_arc/ARC_Easy",
    "kilt_tasks/hotpotqa",
    "trivia_qa/unfiltered",
    "web_questions",
    "wiki_qa",
    "adversarial_qa/dbidaf",
    "adversarial_qa/dbert",
    "adversarial_qa/droberta",
    "duorc/SelfRC",
    "duorc/ParaphraseRC",
    "ropes",
    "squad_v2",
    "quoref",
    # "tydiqa",
    "cos_e/v1.11",
    "cosmos_qa",
    "dream",
    "openbookqa/main",
    "qasc",
    "quail",
    "quarel",
    "quartz",
    "race/high",
    "race/middle",
    "sciq",
    "social_i_qa",
    "wiki_hop/original",
    "wiqa",
    "piqa",
    "amazon_polarity",
    "app_reviews",
    "imdb",
    "rotten_tomatoes",
    "yelp_review_full",
    "hellaswag",
    "common_gen",
    "wiki_bio",
    "cnn_dailymail/3.0.0",
    "gigaword",
    "multi_news",
    "samsum",
    "xsum",
    "ag_news",
    "dbpedia_14",
    "trec",
    "super_glue/multirc",
    "super_glue/wsc.fixed",
    "super_glue/wic",
    "super_glue/copa",
    "super_glue/record",
    "super_glue/boolq",
]

T0_PLUS_PLUS_TRAIN_TASK_LIST=[]
for task_name in T0_PLUS_PLUS_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_PLUS_PLUS_TRAIN_TASK_LIST = T0_PLUS_PLUS_TRAIN_TASK_LIST + sub_list



class P3Config(datasets.BuilderConfig):
    """BuilderConfig for P3."""

    def __init__(self, splits, features_dict, score_eval, **kwargs):
        """BuilderConfig for P3.

        Args:
          splits: `List[str]`, the lists of splits which are available for this task
          features_dict: `dict`, the dict of features for this task
          score_eval: `bool`, whether this is task formulated as a rank classification problem
          **kwargs: keyword arguments forwarded to super.
        """
        # Version history:
        # 0.1 initial commit
        super(P3Config, self).__init__(version=datasets.Version("0.1.0"), **kwargs)
        self.splits = splits
        self.features_dict = features_dict
        self.score_eval = score_eval


class P3(datasets.GeneratorBasedBuilder):
    """Subset of P3 used in `Multitask Prompted Training Enables Zero-Shot Task Generalization`"""

    BUILDER_CONFIGS = [
        P3Config(
            name=task_name,
            splits=splits_and_features_dict["splits"],
            features_dict=splits_and_features_dict["features_dict"],
            score_eval=task_name.endswith("score_eval")
        )
        for task_name, splits_and_features_dict in _TASK_SPLITS_AND_FEATURES_DICT.items()
    ]

    def _info(self):
        # All features available are: 'inputs', 'inputs_pretokenized', 'targets',
        # 'targets_pretokenized', 'idx', 'is_correct', 'weight', and 'answer_choices'
        _FEAT_MAPPING = {
            "answer_choices": datasets.Sequence(datasets.Value("string")),
            "inputs": datasets.Sequence(datasets.Value("int32")),
            "inputs_pretokenized": datasets.Value("string"),
            "targets": datasets.Sequence(datasets.Value("int32")),
            "targets_pretokenized": datasets.Value("string"),
            "idx": datasets.Sequence(datasets.Value("int32")),
            "weight": datasets.Value("float32"),
            "is_correct": datasets.Value("bool"),
        }

        features = {}
        for feat_name in self.config.features_dict.keys():
            features[feat_name] = _FEAT_MAPPING[feat_name]

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(features),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        )

    def _split_generators(self, dl_manager):

        # data_dir = dl_manager.download_and_extract(_URLs)
        data_dir = _URLs
        split_generators = []
        task_name = self.config.name
        if "train" in self.config.splits:
            split_name = "train"
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    gen_kwargs={
                        "tfrecord": data_dir[task_name][split_name]["tfrecord"],
                    }
                )
            )
        if "validation" in self.config.splits:
            split_name = "validation"
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    gen_kwargs={
                        "tfrecord": data_dir[task_name][split_name]["tfrecord"],
                    }
                )
            )
        if "test" in self.config.splits:
            split_name = "test"
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    gen_kwargs={
                        "tfrecord": data_dir[task_name][split_name]["tfrecord"],
                    }
                )
            )
        # Handle splits that are not train, validation or test
        special_splits = set(self.config.splits) - set(["train", "validation", "test"])
        for special_split_name in special_splits:
            split_generators.append(
                datasets.SplitGenerator(
                    name=datasets.Split(special_split_name),
                    gen_kwargs={
                        "tfrecord": data_dir[task_name][special_split_name]["tfrecord"],
                    }
                )
            )
        return split_generators


    def _generate_examples(self, tfrecord):
        """This function returns the examples in the raw (text) form."""
        _FEAT_MAPPING_FUNCTIONS = {
            "answer_choices": lambda x: [choice.decode("utf-8") for choice in x],
            "inputs": lambda x: x.tolist(),
            "inputs_pretokenized": lambda x: x.decode("utf-8"),
            "targets": lambda x: x.tolist(),
            "targets_pretokenized": lambda x: x.decode("utf-8"),
            "idx": lambda x: x.tolist(),
            "weight": lambda x: float(x),
            "is_correct": lambda x: x,
        }

        key = 0
        features_dict = self.config.features_dict
        ds = load_cached_task(features_dict, tfrecord)

        for ex in ds.as_numpy_iterator():
            ex_dict = {}
            for feat_name, feat_value in ex.items():
                ex_dict[feat_name] = _FEAT_MAPPING_FUNCTIONS[feat_name](feat_value)
            yield key, ex_dict
            key += 1
