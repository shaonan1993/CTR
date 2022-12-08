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
import json
from collections import defaultdict

_MAX_DATASET_SIZE=500000
# _RAW_DATA_PATH = "/dataset/fd5061f6/yanan/data/P3"
_RAW_DATA_PATH="/sharefs/english/yanan/data/P3"
# _DATA_PATH = "/dataset/fd5061f6/yanan/data/P3/data"
_DATA_PATH = "/sharefs/english/yanan/data/P3/data"

def find_task_splits_and_features_dict():
    """Get the task available (list was pre-computed by `print_data_split_sizes.py`), and get the features for each task."""
    task_splits_and_features = defaultdict(dict)
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
    return task_splits_and_features


# _TASK_SPLITS_AND_FEATURES_DICT = find_task_splits_and_features_dict()
# P3_TASK_LIST = list(_TASK_SPLITS_AND_FEATURES_DICT.keys())


datasets_without_validation = [
    "ag_news", "dbpedia_14", "trec", "amazon_polarity", "imdb", "yelp_review_full", "wiki_bio",
    "web_questions"]


MY_P3_DIR = './data/my_P3'

P3_TASK_LIST = os.listdir(MY_P3_DIR)


### TODO
cls_template_names = ['ag_news_classify_question_first', 'ag_news_classify_with_choices_question_first',
                      'ag_news_recommend', 'ag_news_which_section_choices', 'ag_news_which_section',
                      'ag_news_classify_with_choices', 'ag_news_classify', 'app_reviews_categorize_rating_using_review',
                      'app_reviews_convert_to_star_rating', 'wiki_hop_original_choose_best_object_interrogative_1',
                      'wiki_hop_original_choose_best_object_affirmative_1',
                      'wiki_hop_original_choose_best_object_affirmative_3',
                      'wiki_hop_original_choose_best_object_affirmative_2',
                      'wiki_hop_original_choose_best_object_interrogative_2',
                      'glue_mrpc_want_to_know', 'glue_mrpc_paraphrase', 'glue_mrpc_equivalent',
                      'glue_mrpc_replace', 'glue_mrpc_same_thing', 'glue_qqp_quora', 'glue_qqp_duplicate_or_not',
                      'glue_qqp_same_thing', 'glue_qqp_answer', 'glue_qqp_meaning', 'glue_qqp_duplicate',
                      'amazon_polarity_Is_this_review', 'amazon_polarity_User_recommend_this_product',
                      'amazon_polarity_Is_this_product_review_positive', 'amazon_polarity_Is_this_review_negative',
                      'amazon_polarity_convey_negative_or_positive_sentiment', 'amazon_polarity_negative_or_positive_tone',
                      'amazon_polarity_user_satisfied', 'amazon_polarity_would_you_buy', 'amazon_polarity_flattering_or_not',
                      'paws_labeled_final_task_description_no_label', 'paws_labeled_final_Meaning',
                      'paws_labeled_final_context_question_no_label', 'paws_labeled_final_Rewrite_no_label',
                      'paws_labeled_final_context_question', 'paws_labeled_final_Concatenation',
                      'paws_labeled_final_Concatenation_no_label', 'paws_labeled_final_Meaning_no_label',
                      'paws_labeled_final_PAWS_ANLI_GPT3', 'paws_labeled_final_Rewrite',
                      'paws_labeled_final_PAWS_ANLI_GPT3_no_label',
                      'dbpedia_14_given_list_what_category_does_the_paragraph_belong_to',
                      'dbpedia_14_pick_one_category_for_the_following_text', 'dbpedia_14_given_a_choice_of_categories_',
                      'dbpedia_14_given_a_list_of_category_what_does_the_title_belong_to', 'dream_baseline',
                      'dream_read_the_following_conversation_and_answer_the_question', 'trec_what_category_best_describe',
                      'trec_fine_grained_ENTY', 'trec_pick_the_best_descriptor', 'trec_fine_grained_open_context_first',
                      'trec_which_category_best_describes', 'trec_trec1', 'trec_trec2', 'trec_fine_grained_open',
                      'imdb_Movie_Expressed_Sentiment_2', 'imdb_Reviewer_Opinion_bad_good_choices', 'imdb_Sentiment_with_choices_',
                      'imdb_Reviewer_Sentiment_Feeling', 'imdb_Writer_Expressed_Sentiment', 'imdb_Movie_Expressed_Sentiment',
                      'imdb_Text_Expressed_Sentiment', 'imdb_Reviewer_Enjoyment_Yes_No', 'imdb_Reviewer_Expressed_Sentiment',
                      'imdb_Reviewer_Enjoyment', 'rotten_tomatoes_Reviewer_Opinion_bad_good_choices', 'rotten_tomatoes_Text_Expressed_Sentiment', 'rotten_tomatoes_Sentiment_with_choices_',
                      'rotten_tomatoes_Reviewer_Enjoyment_Yes_No', 'rotten_tomatoes_Reviewer_Enjoyment', 'rotten_tomatoes_Movie_Expressed_Sentiment',
                      'rotten_tomatoes_Writer_Expressed_Sentiment', 'rotten_tomatoes_Movie_Expressed_Sentiment_2', 'rotten_tomatoes_Reviewer_Expressed_Sentiment',
                      'rotten_tomatoes_Reviewer_Sentiment_Feeling', 'yelp_review_full_so_i_would', 'yelp_review_full_based_on_that', 'yelp_review_full_format_star',
                      'yelp_review_full_this_place', 'yelp_review_full_format_score', 'yelp_review_full_on_a_scale', 'yelp_review_full_format_rating',
                      'wiki_qa_Is_This_True_', 'wiki_qa_automatic_system', 'wiki_qa_found_on_google', 'wiki_qa_exercise', 'wiki_qa_Decide_good_answer',
                      'sciq_Direct_Question_Closed_Book_', 'sciq_Multiple_Choice_Closed_Book_', 'sciq_Multiple_Choice_Question_First', 'sciq_Multiple_Choice',
                      'sciq_Direct_Question', 'quarel_do_not_use', 'quarel_logic_test', 'quarel_heres_a_story', 'quarel_choose_between',
                      'quarel_testing_students', 'qasc_is_correct_1', 'qasc_qa_with_separated_facts_1', 'qasc_qa_with_separated_facts_3',
                      'qasc_qa_with_separated_facts_4', 'qasc_qa_with_separated_facts_5', 'qasc_qa_with_combined_facts_1', 'qasc_is_correct_2',
                      'qasc_qa_with_separated_facts_2', 'cosmos_qa_description_context_question_answer_text', 'cosmos_qa_description_context_question_text',
                      'cosmos_qa_description_context_question_answer_id', 'cosmos_qa_context_description_question_answer_text', 'cosmos_qa_no_prompt_id',
                      'cosmos_qa_context_question_description_text', 'cosmos_qa_no_prompt_text', 'cosmos_qa_context_description_question_answer_id',
                      'cosmos_qa_context_question_description_answer_id', 'cosmos_qa_context_description_question_text',
                      'cosmos_qa_context_question_description_answer_text', 'cosmos_qa_only_question_answer', 'wiqa_effect_with_string_answer',
                      'wiqa_which_of_the_following_is_the_supposed_perturbation', 'wiqa_effect_with_label_answer', 'wiqa_does_the_supposed_perturbation_have_an_effect',
                      'social_i_qa_I_was_wondering', 'social_i_qa_Show_choices_and_generate_answer', 'social_i_qa_Check_if_a_random_answer_is_valid_or_not', 'social_i_qa_Generate_answer',
                      'social_i_qa_Show_choices_and_generate_index', 'quail_context_question_answer_description_id', 'quail_context_question_answer_description_text',
                      'quail_description_context_question_answer_id', 'quail_context_question_description_answer_text', 'quail_context_question_description_text',
                      'quail_context_description_question_text', 'quail_context_question_description_answer_id', 'quail_no_prompt_id', 'quail_context_description_question_answer_id',
                      'quail_description_context_question_text', 'quail_no_prompt_text', 'quail_context_description_question_answer_text', 'quail_description_context_question_answer_text',
                      'quartz_use_info_from_question_paragraph', 'quartz_paragraph_question_plain_concat', 'quartz_use_info_from_paragraph_question',
                      'quartz_answer_question_based_on', 'quartz_answer_question_below',
                      'quartz_read_passage_below_choose', 'quartz_having_read_above_passage', 'quartz_given_the_fact_answer_the_q', 'cos_e_v1.11_question_description_option_text',
                      'cos_e_v1.11_question_description_option_id', 'cos_e_v1.11_question_option_description_text',
                      'cos_e_v1.11_description_question_option_id', 'cos_e_v1.11_description_question_option_text', 'cos_e_v1.11_question_option_description_id',
'trec_fine_grained_LOC', 'trec_fine_grained_NUM_context_first', 'trec_fine_grained_NUM', 'trec_fine_grained_LOC_context_first',
                      'trec_fine_grained_DESC', 'trec_fine_grained_ABBR', 'trec_fine_grained_ABBR_context_first',
                      'trec_fine_grained_HUM', 'trec_fine_grained_HUM_context_first', 'trec_fine_grained_DESC_context_first'
                      ]


gen_template_names = [
    'app_reviews_generate_review',
    'app_reviews_convert_to_rating',
    'wiki_bio_who',
    'wiki_bio_comprehension',
    'wiki_bio_what_content',
    'wiki_bio_guess_person',
    'wiki_bio_key_content',
    'cnn_dailymail_3.0.0_write_an_outline',
    'cnn_dailymail_3.0.0_news_summary',
    'cnn_dailymail_3.0.0_2_or_3_sentences',
    'cnn_dailymail_3.0.0_tldr_summary',
    'cnn_dailymail_3.0.0_news_card_view',
    'cnn_dailymail_3.0.0_generate_story',
    'cnn_dailymail_3.0.0_sum_in_brief',
    'cnn_dailymail_3.0.0_news_stock',
    'cnn_dailymail_3.0.0_spice_up_story',
    'gigaword_generate_summary_for_this',
    'gigaword_reverse_writing',
    'gigaword_make_a_title',
    'gigaword_first_sentence_title',
    'gigaword_TLDR',
    'gigaword_write_its_sentence',
    'gigaword_write_a_title_for_this_sentence',
    'gigaword_in_a_nutshell',
    'gigaword_write_an_article',
    'wiki_hop_original_explain_relation',
    'wiki_hop_original_generate_object',
    'wiki_hop_original_generate_subject',
    'wiki_hop_original_generate_subject_and_object',
    'glue_mrpc_generate_paraphrase', 'glue_mrpc_generate_sentence',
    'paws_labeled_final_paraphrase_task',
    'dream_generate_last_utterance', 'dream_answer_to_dialogue', 'dream_generate_first_utterance',
    'kilt_tasks_hotpotqa_complex_question', 'kilt_tasks_hotpotqa_combining_facts', 'kilt_tasks_hotpotqa_formulate',
    'kilt_tasks_hotpotqa_final_exam', 'kilt_tasks_hotpotqa_straighforward_qa',
    'multi_news_what_are_the_key_points', 'multi_news_synthesize', 'multi_news_summary_scenario',
    'multi_news_summarize', 'multi_news_expand_reverse_task_', 'multi_news_distill',
    'samsum_Summarize_this_dialogue_', 'samsum_Given_the_above_dialogue_write_a_summary',
    'samsum_Summarize_', 'samsum_To_sum_up_this_dialog', 'samsum_Generate_a_summary_for_this_dialogue',
    'samsum_Write_a_dialogue_that_match_this_summary', 'samsum_Sum_up_the_following_dialogue',
    'xsum_DOC_write_summary_of_above', 'xsum_article_DOC_summary', 'xsum_DOC_how_would_you_rephrase_few_words',
    'xsum_college_roommate_asked_DOC_so_I_recap', 'xsum_DOC_boils_down_to_simple_idea_that', 'xsum_summarize_DOC',
    'xsum_summarize_this_DOC_summary', 'xsum_DOC_given_above_write_one_sentence', 'xsum_read_below_DOC_write_abstract',
    'xsum_DOC_tldr', 'imdb_Negation_template_for_positive_and_negative', 'wiki_qa_Jeopardy_style',
    'wiki_qa_Topic_Prediction_Question_and_Answer_Pair', 'wiki_qa_Generate_Question_from_Topic',
    'wiki_qa_Topic_Prediction_Question_Only', 'wiki_qa_Topic_Prediction_Answer_Only',
    'wiki_qa_Direct_Answer_to_Question', 'common_gen_Given_concepts_type_2',
    'common_gen_Put_together', 'common_gen_choice_in_concept_centric_sentence_generation',
    'common_gen_random_task_template_prompt', 'common_gen_topics_from_the_sentence', 'common_gen_sentence_to_concepts',
    'common_gen_topic_to_sentence', 'common_gen_Example_prompt', 'common_gen_Given_concepts_type_1',
    'adversarial_qa_dbidaf_based_on', 'adversarial_qa_dbidaf_answer_the_following_q',
    'adversarial_qa_dbidaf_generate_question', 'adversarial_qa_dbidaf_tell_what_it_is',
    'adversarial_qa_dbidaf_question_context_answer', 'adversarial_qa_dbert_generate_question',
    'adversarial_qa_dbert_tell_what_it_is', 'adversarial_qa_dbert_question_context_answer',
    'adversarial_qa_dbert_based_on', 'adversarial_qa_dbert_answer_the_following_q',
    'adversarial_qa_droberta_generate_question', 'adversarial_qa_droberta_tell_what_it_is',
    'adversarial_qa_droberta_question_context_answer', 'adversarial_qa_droberta_based_on',
    'adversarial_qa_droberta_answer_the_following_q', 'quoref_Guess_Answer', 'quoref_Answer_Question_Given_Context',
    'quoref_Find_Answer', 'quoref_Context_Contains_Answer', 'quoref_Given_Context_Answer_Question',
    'quoref_What_Is_The_Answer', 'quoref_Answer_Test', 'quoref_Guess_Title_For_Context', 'quoref_Found_Context_Online',
    'quoref_Answer_Friend_Question', 'quoref_Read_And_Extract_', 'ropes_prompt_beginning',
    'ropes_prompt_bottom_no_hint', 'ropes_prompt_bottom_hint_beginning', 'ropes_given_background_situation',
    'ropes_plain_no_background', 'ropes_plain_bottom_hint', 'ropes_plain_background_situation',
    'ropes_background_new_situation_answer', 'ropes_background_situation_middle', 'ropes_new_situation_background_answer',
    'ropes_prompt_mix', 'ropes_read_background_situation', 'duorc_SelfRC_generate_question_by_answer',
    'duorc_SelfRC_movie_director', 'duorc_SelfRC_extract_answer', 'duorc_SelfRC_generate_question',
    'duorc_SelfRC_answer_question', 'duorc_SelfRC_build_story_around_qa', 'duorc_SelfRC_question_answering',
    'duorc_SelfRC_title_generation', 'duorc_SelfRC_decide_worth_it', 'duorc_ParaphraseRC_build_story_around_qa',
    'duorc_ParaphraseRC_decide_worth_it', 'duorc_ParaphraseRC_question_answering', 'duorc_ParaphraseRC_movie_director',
    'duorc_ParaphraseRC_generate_question', 'duorc_ParaphraseRC_extract_answer', 'duorc_ParaphraseRC_title_generation',
    'duorc_ParaphraseRC_answer_question', 'duorc_ParaphraseRC_generate_question_by_answer', 'cosmos_qa_context_answer_to_question',
    'wiqa_what_might_be_the_first_step_of_the_process', 'wiqa_what_might_be_the_last_step_of_the_process',
    'wiqa_what_is_the_missing_first_step', 'wiqa_what_is_the_final_step_of_the_following_process',
    'social_i_qa_Generate_the_question_from_the_answer', 'cos_e_v1.11_rationale', 'cos_e_v1.11_aligned_with_common_sense',
    'cos_e_v1.11_explain_why_human', 'cos_e_v1.11_generate_explanation_given_text', 'cos_e_v1.11_i_think'
]





"""
large_t0_tasks = ["gigaword", "amazon_polarity", "wiki_bio", "dbpedia_14", "yelp_review_full"]
large_t0_tasks_prompt_count = {}
for large_task in large_t0_tasks:
    for task_name in P3_TASK_LIST:
        if task_name.startswith(large_task):
            if large_task not in large_t0_tasks_prompt_count:
                large_t0_tasks_prompt_count[large_task] = 1
            else:
                large_t0_tasks_prompt_count[large_task] += 1
"""
"""
large_t0_task_dict = {}
for cur_task in P3_TASK_LIST:
    for large_task_prefix in large_t0_tasks_prompt_count.keys():
        if cur_task.startswith(large_task_prefix):
            large_t0_task_dict[cur_task] = int(_MAX_DATASET_SIZE / large_t0_tasks_prompt_count[large_task_prefix])
"""

DEBUG_TRAIN_TASK_NAME = ["ropes"]
DEBUG_TRAIN_TASK_LIST=[]
for task_name in DEBUG_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    DEBUG_TRAIN_TASK_LIST = DEBUG_TRAIN_TASK_LIST + sub_list

T0_SUB_TRAIN_TASK_NAME = ['commonsense_qa', 'cosmos_qa', 'cos_e/v1.11', 'qasc', 'super_glue/boolq', 'rotten_tomatoes',
                          'ag_news', 'glue/mrpc', 'glue/qqp']
T0_SUB_TRAIN_TASK_LIST = []
for task_name in T0_SUB_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_SUB_TRAIN_TASK_LIST = T0_SUB_TRAIN_TASK_LIST + sub_list

T0_TRAIN_TASK_NAME = [
    "ag_news", ####1
    "app_reviews", ####1
    "wiki_bio", #2
    "cnn_dailymail/3.0.0",#2
    "gigaword",#2
    "wiki_hop/original", #2
    "glue/mrpc",####1
    "glue/qqp",####1
    "amazon_polarity",####1
    "paws/labeled_final",####1
    "dbpedia_14",#2
    "dream",####1
    "kilt_tasks/hotpotqa",#2
    "trec",####1
    "multi_news",#2
    "samsum",#2
    "xsum",#2
    "imdb",#2
    "rotten_tomatoes",####1
    "yelp_review_full",####1
    "wiki_qa",####1
    "common_gen",#2
    "adversarial_qa/dbidaf", #2
    "adversarial_qa/dbert", #2
    "adversarial_qa/droberta",#2
    "quoref",#2
    "ropes",#2
    "duorc/SelfRC",#2
    "duorc/ParaphraseRC",#2
    "sciq",####1
    "quarel",####1
    "qasc",####1
    "cosmos_qa",####1
    "wiqa",####1
    "social_i_qa",####1
    "quail", #2
    "quartz",####1
    "cos_e/v1.11", ####1
    "commonsense_qa"
]
T0_TRAIN_TASK_LIST=[]
for task_name in T0_TRAIN_TASK_NAME:
    task_name = task_name.replace("/", "_")
    sub_list = [task_li for task_li in P3_TASK_LIST if task_li.startswith(task_name)]
    T0_TRAIN_TASK_LIST = T0_TRAIN_TASK_LIST + sub_list

T0_TRAIN_MINIMAL_TASK_LIST = []


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
