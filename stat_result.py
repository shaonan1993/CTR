"""
state eval result
"""
import os

if __name__ == '__main__':
    result_dir = './eval_results/t0-large'

    result_list = os.listdir(result_dir)
    result_list = [name for name in result_list if name.endswith('-results.txt')]
    print(result_list)
    result_dict = {}  # task_name -> result

    for result_file_name in result_list:
        for line in open(os.path.join(result_dir, result_file_name), 'r').readlines():
            if line.startswith('median:'):
                median = line.split(':')[1]
                median = median[:-2]
                median = median.strip()
                median = float(median)
                median *= 100

            if line.startswith('mean:'):
                mean = line.split(':')[1]
                mean = mean[:-2]
                mean = mean.strip()
                mean = float(mean)
                mean *= 100
        task_name = result_file_name.replace('-results.txt', '')
        result_dict[task_name] = {'median': median, 'mean': mean}

    # print(result_dict)
    if 'anli_r1' in result_dict:
        value = result_dict['anli_r1']
        print(f'anli_r1\t{value["mean"]}\t{value["median"]}')
    if 'anli_r2' in result_dict:
        value = result_dict['anli_r2']
        print(f'anli_r2\t{value["mean"]}\t{value["median"]}')
    if 'anli_r3' in result_dict:
        value = result_dict['anli_r3']
        print(f'anli_r3\t{value["mean"]}\t{value["median"]}')
    if 'super_glue_cb' in result_dict:
        value = result_dict['super_glue_cb']
        print(f'super_glue_cb\t{value["mean"]}\t{value["median"]}')
    if 'super_glue_rte' in result_dict:
        value = result_dict['super_glue_rte']
        print(f'super_glue_rte\t{value["mean"]}\t{value["median"]}')
    if 'super_glue_wsc.fixed' in result_dict:
        value = result_dict['super_glue_wsc.fixed']
        print(f'super_glue_wsc.fixed\t{value["mean"]}\t{value["median"]}')
    if 'winogrande_winogrande_xl' in result_dict:
        value = result_dict['winogrande_winogrande_xl']
        print(f'winogrande_winogrande_xl\t{value["mean"]}\t{value["median"]}')
    if 'super_glue_copa' in result_dict:
        value = result_dict['super_glue_copa']
        print(f'super_glue_copa\t{value["mean"]}\t{value["median"]}')
    if 'hellaswag' in result_dict:
        value = result_dict['hellaswag']
        print(f'hellaswag\t{value["mean"]}\t{value["median"]}')
    if 'super_glue_wic' in result_dict:
        value = result_dict['super_glue_wic']
        print(f'super_glue_wic\t{value["mean"]}\t{value["median"]}')

    # for key, value in result_dict.items():
    #     print(f'{key}\t{value["mean"]}\t{value["median"]}')
    sum_score = sum([value["mean"] for value in result_dict.values()])
    print(f'all mean: {sum_score / len(result_dict)}')

