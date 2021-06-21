import torch
import argparse

parser = argparse.ArgumentParser(description='Read results and store them..')


parser.add_argument('--tasks', type=int, default=20, help='Number of tasks')
parser.add_argument('text_file', type=str, help="File of results")
parser.add_argument('res_name', type=str, help="File to store results")
parser.add_argument('name', type=str, help='Name of method')

args = parser.parse_args()


res_name = args.res_name
text_file = args.text_file
n_tasks = args.tasks

R = torch.zeros([n_tasks, n_tasks])
tr_task, te_task = 0, 0
with open(text_file, 'r') as f:
    line = f.readline()
    while line:
      if 'Accuracies' in line or 'Average' in line:
          line = f.readline()
          continue
      line = float(line.split(' ')[-2])
      R[tr_task, te_task] = line
      print("R[%d, %d] = %.2f" % (tr_task, te_task, line))
      te_task += 1
      if te_task > n_tasks - 1:
          te_task = 0
          tr_task += 1
      line = f.readline()

accs = {'Task ' + str(n): R[-1, n].item() for n in range(n_tasks)}
avg_acc = sum(accs.values()) / len(accs)
try:
    results = torch.load(res_name)
except Exception as e:
    logger.warning('Exception: %s' % str(e))
    results = {}
accs['Average'] = avg_acc
accs['R'] = R
results[args.name] = accs
torch.save(results, res_name)
