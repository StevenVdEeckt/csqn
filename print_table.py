import torch
import sys
import argparse
import logging
import numpy as np


parser = argparse.ArgumentParser(description='Print the results..')

parser.add_argument('file_name', help='Name of the results file', type=str)
parser.add_argument('--start', help='First task to consider - default is 0 (= first task)', type=int, default=0)
parser.add_argument('--end', help='Final task to consider - default is -1 (= last task)', type=int, default=-1)
parser.add_argument('--decimals', help='Decimals for rounding - default is 2', type=int, default=2)
parser.add_argument('--fine_tuning', type=str, default="Fine-Tuning",
                    help='Name of Fine-Tuning baselines (used to compute forward transfer) - default is Fine-Tuning')
parser.add_argument('--log_file', type=str, default="FALSE",
                    help='File to write the output to - if FALSE (default), writing to Terminal')


args = parser.parse_args()


FORMAT = '%(message)s'

if args.log_file == 'FALSE':
    logging.basicConfig(level=logging.INFO, format=FORMAT)
else:
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG, format=FORMAT)
logger = logging.getLogger('main')


""" Loading the parameters from parser """
name = args.file_name  # file where to results are stored
start, end = args.start, args.end  # first and last task
bit = args.decimals  # for rounding

""" Loading the results dictionary """
try:
    res = torch.load(name, map_location='cpu')
except Exception as e:
    raise Exception("Error: " + str(e) + " - give a valid file name: ", name)


if end == -1:
  end = res[list(res.keys())[0]]['R'].size(0)  # if end == -1, set end to last task

fine_tuning = args.fine_tuning  # name of Fine-Tuning baselines, used for computing forward transfer

"""
    Computes accuracy, backward transfer and forward transfer for a method based on its R matrix
    :param torch.tensor R: 2D matrix where R[i,j] is the accuracy on task j after training task i
    :param int start: first task to consider
    :param int end: last task to consider
    :param torch.tensor Rft: (optional) 2D matrix R for Fine-Tuning - used to compute forward transfer
"""
def compute_acc_bwt_fwt(R, start, end, Rft=None):
    res_dict = {}
    result = [round(R[end-1, i].item(), bit) for i in range(start, end)]
    res_dict['result'] = result
    avg = round(sum(result) / len(result), bit)
    res_dict['avg'] = avg
    bwt, fwt = 0, 0
    for i in range(start, end):
        bwt += (R[end-1, i].item() - R[i,i].item())
        if Rft is not None:
            fwt += (R[i,i].item() - Rft[i, i].item())
    bwt, fwt = round(bwt / (end - 1), bit) if end > 1 else None, round(fwt / end, bit) if Rft is not None else None
    res_dict['fwt'] = fwt if fwt is not None else 'NA'
    res_dict['bwt'] = bwt if bwt is not None else 'NA'
    return res_dict


"""
    Compute overview dictionary, which contains measures such as average accuracy, backward transfer and 
    forward transfer for each method
    :param dict res: results dictionary where keys are methods and each method has an object 'R' which is its R matrix
    :param int start: first task to consider
    :param int end: last task to consider
"""
def compute_overview(res, start, end):
    overview = {}
    Rft = res[fine_tuning]['R'] if fine_tuning in res.keys() else None
    for n, p in res.items():
        overview[n] = compute_acc_bwt_fwt(p['R'], start, end, Rft)
    return overview


""" Computing the overview """
overview = compute_overview(res=res, start=start, end=end)


"""
    Table 1: contains the results per task after training task 'end'
"""
logger.info("TABLE 1: FINAL RESULTS PER RUN")
row_format ="{:>35}" + "{:>7}" * (end-start)
top = ['T%d' % (i) for i in range(start, end)]
logger.info(row_format.format("", *top))
for team, row in overview.items():
  logger.info(row_format.format(team,  *row['result']))


logger.info('')
logger.info('')

"""
    Table 2: prints a summary per model, i.e. its average accuracy, backward transfer, forward transfer
"""
logger.info("TABLE 2: SUMMARY PER RUN")
row_format ="{:>35}" + "{:>15}"  * (3)
top = ['Average', 'BWT', 'FWT']
logger.info(row_format.format("", *top))
for team, row in overview.items():
  logger.info(row_format.format(team, row['avg'], row['bwt'], row['fwt']))

"""
    Returns a dictionary with summary results for each CL method, i.e. averaged over the multiple runs
"""
def compute_average(overview):
    average = {}
    get_name = lambda x: ' '.join(x.split(' ')[:-1]) # strip the ID off of the name
    for n, p in overview.items():
        try:
            int(n.split(' ')[-1])
            name = get_name(n)
        except Exception as e:
            name = n
        try:
            average[name]['k'] += 1
            average[name]['avg'] = average[name]['avg'] * (average[name]['k'] - 1) / average[name]['k'] + p['avg'] / \
                                   average[name]['k']
            if p['bwt'] != 'NA':
                average[name]['bwt'] = average[name]['bwt'] * (average[name]['k'] - 1) / average[name]['k'] + p['bwt'] / \
                                   average[name]['k']
            if p['fwt'] != 'NA':
                average[name]['fwt'] = average[name]['fwt'] * (average[name]['k'] - 1) / average[name]['k'] + p['fwt'] / \
                                   average[name]['k']
        except Exception as e:
            average[name] = {'avg': p['avg'], 'bwt': p['bwt'], 'fwt': p['fwt'], 'k': 1}
    for n, p in overview.items():
        try:
            int(n.split(' ')[-1])
            name = get_name(n)
            try:
                average[name]['std'] += (p['avg'] - average[name]['avg']) ** 2 / (average[name]['k'] - 1)
            except:
                average[name]['std'] = (p['avg'] - average[name]['avg']) ** 2 / (average[name]['k'] - 1)
        except:
            continue
    for n, p in average.items():
        if 'std' in p.keys():
            p['std'] = np.sqrt(p['std'])
    return average

""" Computing the average per CL method """
average = compute_average(overview)

""" Displaying number of runs per method """
for n, p in average.items():
   logger.info(n + " had %d runs" % p['k'])


logger.info('')
logger.info('')

_round = lambda x, bit: round(x, bit) if x != 'NA' else 'NA'


"""
    Table 3: prints a summary per CL method (averaged over the number of runs)
"""
logger.info("TABLE 3: SUMMARY PER CL METHOD")
row_format ="{:>35}" + "{:>20}" +  "{:>15}" * (2)
top = ['Average', 'BWT', 'FWT']
logger.info(row_format.format("", *top))
for team, row in average.items():
    if 'std' in row.keys():
        logger.info(row_format.format(team, str(_round(row['avg'], bit)) + " ± " + str(_round(row['std'], bit)),
                                _round(row['bwt'], bit), _round(row['fwt'], bit)))
    else:
        logger.info(row_format.format(team, _round(row['avg'], bit), _round(row['bwt'], bit), _round(row['fwt'], bit)))


