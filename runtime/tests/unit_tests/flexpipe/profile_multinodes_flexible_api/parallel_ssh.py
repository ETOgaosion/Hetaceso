import time
from pssh.clients import ParallelSSHClient, SSHClient
import pprint
import math
import signal
import sys

'''
Hint: Modify these Configurations only
All functions are extendable
'''
gpus_per_nodes = 4
required_nodes = 2
test_nums = [0]

hosts = ['localhost', '10.156.154.242']
# hosts = ['localhost']
localhost_ip = '10.156.154.20'
host_project_dirs = {'localhost': '/home/gzy/projects/Hetaceso', '10.156.154.242': '/home/gaoziyuan/projects/Hetaceso'}
container_project_dir = '/workspace/Hetaceso'
contianer_names = {'localhost': 'hetaceso-gzy', '10.156.154.242': 'hetaceso-gaoziyuan'}
pwd_relative = 'runtime/tests/unit_tests/flexpipe/profile_multinodes_flexible_api'
users = {'localhost': 'gzy', '10.156.154.242': 'gaoziyuan'}
password = 'gaoziyuan'
pkey = '/home/gzy/.ssh/id_rsa'


'''
input check
'''
# assert len(required_nodes) == len(required_data_parallel_size) == len(required_micro_batch_size)


'''
preparations
'''
localhost = 'localhost'
localhostclient = SSHClient(localhost, pkey=pkey, user=users['localhost'], password=password)

seperate_clients_hosts = {}
for host in hosts:
    seperate_clients_hosts[host] = ParallelSSHClient([host], pkey=pkey, user=users[host], password=password)

def preparation():
    for host in hosts:
        output_git_pull = seperate_clients_hosts[host].run_command('cd ' + host_project_dirs[host] + ' && git pull origin main')
        seperate_clients_hosts[host].join(output_git_pull)

# preparation()

def kill_all():
    for host in seperate_clients_hosts:
        print(host)
        output = host.run_command('ps aux | grep python | grep -v grep | awk "{print \$2}" | sudo xargs kill -9 ', sudo=True)
        # for host_out in output:
        #     host_out.stdin.write('gzy2024\n')
        #     host_out.stdin.flush()
        host.join(output)
        for line in output.stdout:
            print(line)
        for line in output.stderr:
            print(line)
    
# kill_all()

def signal_handler(sig, frame):
    kill_all()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

clients = {}
for host in hosts:
    clients[host] = SSHClient(host, pkey=pkey, user=users[host], password=password)

'''
Preparation of clients and commands
'''
all_commands = {}

for test in test_nums:
    all_commands[test] = {}
    for idx, host in enumerate(hosts):
        all_commands[host] = f'docker exec -it {contianer_names[localhost]} bash -c "cd {container_project_dir}/{pwd_relative} && ./run_rank_{idx}.sh {idx} {test}"'
        
pprint.pp(all_commands)


'''
Execution of commands
'''
def execute_command(test_num):
    output = []
    print(f'execute test-{test_num}')
    for k, host in enumerate(hosts):
        output.append(clients[host].run_command(all_commands[test_num][host]))
    for k, host in enumerate(hosts):
        clients[host].wait_finished(output[k])
    for out in output:
        for line in out.stdout:
            print(line)
        for line in out.stderr:
            print(line)
    print('Finish test-{test_num}')

# execute_command(0)

# for nodes in required_nodes:
#     execute_command(nodes)

# kill_all()