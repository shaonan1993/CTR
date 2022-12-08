import os

import datasets
from datasets import load_dataset
from datasets import Dataset

from templates import DatasetTemplates
import codecs
import json


def special_for_triviaqa(uniq_task_dir):
    all_raw_dataset = {}
    if os.path.exists(os.path.join(uniq_task_dir, 'train.json')):
        train_data_dict = {'question': [], 'answer': []}
        with open(os.path.join(uniq_task_dir, 'train.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                train_data_dict['question'].append(example['question'])
                train_data_dict['answer'].append(example['answer'])

        train_dataset = Dataset.from_dict(train_data_dict)
        all_raw_dataset['train'] = train_dataset

    if os.path.exists(os.path.join(uniq_task_dir, 'validation.json')):
        dev_data_dict = {'question': [], 'answer': []}
        with open(os.path.join(uniq_task_dir, 'validation.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                dev_data_dict['question'].append(example['question'])
                dev_data_dict['answer'].append(example['answer'])

        dev_dataset = Dataset.from_dict(dev_data_dict)
        all_raw_dataset['validation'] = dev_dataset
    if os.path.exists(os.path.join(uniq_task_dir, 'test.json')):
        test_data_dict = {'question': [], 'answer': []}
        with open(os.path.join(uniq_task_dir, 'test.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                test_data_dict['question'].append(example['question'])
                test_data_dict['answer'].append(example['answer'])

        test_dataset = Dataset.from_dict(test_data_dict)
        all_raw_dataset['test'] = test_dataset

    return all_raw_dataset


def special_for_wikihop(uniq_task_dir):
    all_raw_dataset = {}
    if os.path.exists(os.path.join(uniq_task_dir, 'train.json')):
        train_data_dict = {'id': [], 'question': [], 'answer': [], 'candidates': [], 'supports': []}
        with open(os.path.join(uniq_task_dir, 'train.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                train_data_dict['id'].append(example['id'])
                train_data_dict['question'].append(example['question'])
                train_data_dict['answer'].append(example['answer'])
                train_data_dict['candidates'].append(example['candidates'])
                train_data_dict['supports'].append(example['supports'])

        train_dataset = Dataset.from_dict(train_data_dict)
        all_raw_dataset['train'] = train_dataset

    if os.path.exists(os.path.join(uniq_task_dir, 'validation.json')):
        dev_data_dict = {'id': [], 'question': [], 'answer': [], 'candidates': [], 'supports': []}
        with open(os.path.join(uniq_task_dir, 'validation.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                dev_data_dict['id'].append(example['id'])
                dev_data_dict['question'].append(example['question'])
                dev_data_dict['answer'].append(example['answer'])
                dev_data_dict['candidates'].append(example['candidates'])
                dev_data_dict['supports'].append(example['supports'])

        dev_dataset = Dataset.from_dict(dev_data_dict)
        all_raw_dataset['validation'] = dev_dataset
    if os.path.exists(os.path.join(uniq_task_dir, 'test.json')):
        test_data_dict = {'id': [], 'question': [], 'answer': [], 'candidates': [], 'supports': []}
        with open(os.path.join(uniq_task_dir, 'test.json')) as f:
            for line in f.readlines():
                example = json.loads(line)
                test_data_dict['id'].append(example['id'])
                test_data_dict['question'].append(example['question'])
                test_data_dict['answer'].append(example['answer'])
                test_data_dict['candidates'].append(example['candidates'])
                test_data_dict['supports'].append(example['supports'])

        test_dataset = Dataset.from_dict(test_data_dict)
        all_raw_dataset['test'] = test_dataset

    return all_raw_dataset


def special_for_duplicat(dataset_config_name, raw_dataset):
    """"remove duplicate example"""
    context_key = 'inputs_pretokenized'

    counter = {}
    for example in raw_dataset:
        if example[context_key] in counter:
            counter[example[context_key]] += 1
        else:
            counter[example[context_key]] = 1

    choice_count = {}
    for key, value in counter.items():
        if value not in choice_count:
            choice_count[value] = 0
        choice_count[value] += 1

    valid_choice_num = sorted(choice_count.items(), key=lambda x: x[1], reverse=True)[0][0]

    valid_example_set = set()
    for key, value in counter.items():
        if value == valid_choice_num:
            valid_example_set.add(key)
        else:
            print(f'duplicate example: {key}')

    # just drop
    raw_dataset = raw_dataset.filter(lambda x: x[context_key] in valid_example_set)

    return raw_dataset


def filter_invalid_data(uniq_task_name, template_name, raw_dataset):
    filtered_dataset = raw_dataset
    if uniq_task_name == 'super_glue_copa':
        if template_name in ["\u2026What could happen next, C1 or C2?", "\u2026As a result, C1 or C2?"]:
            filtered_dataset = raw_dataset.filter(lambda example: example['question'] == 'effect')
        if template_name in ["\u2026which may be caused by", "\u2026why? C1 or C2"]:
            filtered_dataset = raw_dataset.filter(lambda example: example['question'] == 'cause')

    return filtered_dataset


def get_uniq_task_name(dataset_name, subset_name=None, template_name=None):
    uniq_task_name = dataset_name
    if subset_name is not None:
        uniq_task_name += f'_{subset_name}'

    if template_name is not None:
        template_name = template_name.replace('\\', '_')
        template_name = template_name.replace('-', '_')
        template_name = template_name.replace('?', '_')
        template_name = '_'.join(template_name.split())

        uniq_task_name += f'_{template_name}'

    return uniq_task_name


def preprocess_for_P3():
    list_filename = './utils/all.list'

    origin_data_dir = './data/T0_dataset'
    template_dir = './templates_no_prompt'

    output_dir = './data/my_P3_no_prompt'

    datasets.disable_progress_bar()

    os.makedirs(output_dir, exist_ok=True)

    dataset_list = []
    with codecs.open(list_filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = line.replace('\n', '')
            task_tuple = line.split('/')
            if len(task_tuple) == 2:
                dataset_list.append(task_tuple)
            else:
                dataset_list.append((task_tuple[0], None))

    print(dataset_list)

    # preprocess each dataset
    for dataset_name, subset_name in dataset_list:
        uniq_task_name = dataset_name if subset_name is None else f'{dataset_name}_{subset_name}'
        uniq_task_dir = os.path.join(origin_data_dir, uniq_task_name)
        # load dataset
        data_files = {}
        if os.path.exists(os.path.join(uniq_task_dir, 'train.json')):
            data_files['train'] = os.path.join(uniq_task_dir, 'train.json')
        if os.path.exists(os.path.join(uniq_task_dir, 'validation.json')):
            data_files['validation'] = os.path.join(uniq_task_dir, 'validation.json')
        if os.path.exists(os.path.join(uniq_task_dir, 'test.json')):
            if 'validation' not in data_files:
                data_files['test'] = os.path.join(uniq_task_dir, 'test.json')

        print(f'loading dataset: {uniq_task_name}')
        if dataset_name.startswith('trivia'):
            raw_dataset = special_for_triviaqa(uniq_task_dir)
        elif dataset_name.startswith('wiki_hop'):
            raw_dataset = special_for_wikihop(uniq_task_dir)
        else:
            raw_dataset = load_dataset('json', data_files=data_files)

        # fix bug in trec
        if dataset_name == 'trec':
            for split in raw_dataset:
                raw_dataset[split] = raw_dataset[split].rename_column('label-fine', 'label_fine')
                raw_dataset[split] = raw_dataset[split].rename_column('label-coarse', 'label_coarse')

        column_names = raw_dataset['train'].column_names

        def preprocess_train(examples):
            bs = len(examples[column_names[0]])

            input_texts = []
            target_texts = []
            for i in range(bs):
                ex = {
                    k: examples[k][i]
                    for k in column_names
                }
                inputs_and_targets = template.apply(ex)
                if len(inputs_and_targets) == 2:
                    input, target = inputs_and_targets
                else:
                    input, target = '', ''

                input_texts.append(input)
                target_texts.append(target)

            model_inputs = {'inputs_pretokenized': input_texts,
                            'targets_pretokenized': target_texts}

            return model_inputs

        # load template list
        if dataset_name == 'anli':
            prompts = DatasetTemplates(dataset_name, template_dir=template_dir)
        else:
            prompts = DatasetTemplates(dataset_name if subset_name is None else f"{dataset_name}/{subset_name}",
                                       template_dir=template_dir)
        template_list = prompts.templates.keys()
        print(f'{dataset_name}/{subset_name} contains template：{template_list}')

        for template_id in template_list:
            template = prompts.templates[template_id]
            template_name = template.name
            origin_template_name = template.name
            # process for template name
            template_name = template_name.replace('\\', '_')
            template_name = template_name.replace('-', '_')
            template_name = template_name.replace('?', '_')
            template_name = '_'.join(template_name.split())

            prompted_task_name = f'{uniq_task_name}_{template_name}'

            prompted_output_dir = os.path.join(output_dir, prompted_task_name)
            os.makedirs(prompted_output_dir, exist_ok=True)

            # train/validation/test
            for split in raw_dataset:

                print(f'processing prompted_task_name: {prompted_task_name}, split: {split}')
                split_dataset = raw_dataset[split]
                split_dataset = filter_invalid_data(uniq_task_name, origin_template_name, split_dataset)

                if dataset_name == 'trec' or dataset_name.startswith('wiki_qa'):
                    # bug fix
                    preprocessed_split_dataset = split_dataset.map(preprocess_train, batched=True,
                                                                   remove_columns=column_names,
                                                                   load_from_cache_file=False)
                else:
                    preprocessed_split_dataset = split_dataset.map(preprocess_train, batched=True,
                                                                   remove_columns=column_names, num_proc=64,
                                                                   load_from_cache_file=False)
                print(f'after process, contains columns: {preprocessed_split_dataset.column_names}')
                with codecs.open(os.path.join(prompted_output_dir, f'{split}.json'), 'w',
                                 encoding='utf-8') as f:
                    for example in preprocessed_split_dataset:
                        if len(example['inputs_pretokenized']) <= 1 or len(example['targets_pretokenized']) == 0:
                            continue
                        f.write(json.dumps(example) + '\n')


if __name__ == '__main__':
    preprocess_for_P3()

