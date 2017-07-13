import sys
import itertools
from datetime import datetime
from os import mkdir, path, environ
from datetime.relativedelta import relativedelta

args = dict()
get_results_only = False
switch_gpus = False #For multiple GPUs
now = datetime.now()

args['hyper_params'] = ['dataset', 'batch_size', 'hidden', 'dropout']
args['time_step'] = "{} | {} | {} | {}".format(now.month, now.day, now.hour, now.minute)
args['dataset'] = ['delicious']
args['batch_size'] = [10, 25, 50, 100]
args['hidden'] = [1000]
args['dropout'] = [0.5, 0.2, 0.8]

if not get_results_only :
    def time_diff(t1, t2):
        diff = relativedelta(t1, t2)
        return "H: {} | M : {} | S : {}".format(t.hours, t.minutes, t.seconds)

    args_path = '../args'
    if not path.exists(args_path):
        mkdir(args_path)
    np.save(path.join(args_path, args['timestamp']), args)

    stdout_dump_path = '../stdout'
    if not path.exists(stdout_dump_path ):
        mkdir(stdout_dump_path)

    param_values = []
    this_module = sys.modules[__name__]
    for hp_name in args['hyper_params']:
        param_values.append(args[hp_name])
    combinations = list(itertools.product(*param_values))
    n_combinations = len(combinations)
    print('Total no of experiments: ', n_combinations)

    pids = [None] * n_combinations
    f = [None] * n_combinations
    last_process = False
    for i, setting in enumerate(combinations):
        #Create command
        command = "python __main__.py "
        folder_suffix = args['timestamp']
        for name, value in zip(args['hyper_params'], setting):
            command += "--" + name + " " + str(value) + " "
            if name != 'dataset':
                folder_suffix += "_"+str(value)
        command += "--" + "folder_suffix " + folder_suffix
        print(i+1, '/', n_combinations, command)

    if switch_gpus and (i % 2) == 0:
            env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"})
        else:
            env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"})

    name = path.join(stdout_dump_path, folder_suffix)
    with open(name, 'w') as f[i]:
            pids[i] = subprocess.Popen(command.split(), env=env, stdout=f[i])
        if i == n_combinations-1:
            last_process = True
        if ((i+1) % n_parallel_threads == 0 and i >= n_parallel_threads-1) or last_process:
            if last_process and not ((i+1) % n_parallel_threads) == 0:
                n_parallel_threads = (i+1) % n_parallel_threads
            start = datetime.now()
            print('########## Waiting #############')
            for t in range(n_parallel_threads-1, -1, -1):
                pids[i-t].wait()
            end = datetime.now()
            print('########## Waiting Over######### Took', diff(end, start), 'for', n_parallel_threads, 'threads')

    # tabulate_results.write_results(args)