from pssh.clients import ParallelSSHClient, SSHClient
import pprint
import signal
import sys

gpus_per_nodes = 4

hosts = ['10.156.154.242', '10.156.154.20']
# hosts = ['localhost', '10.156.154.20', '10.30.16.24']
localhost_ip = '10.156.154.242'
port = 2230
project_dir = '/workspace/Hetaceso'
user = 'gzy'
password = 'gaoziyuan'
pkey = '/root/.ssh/id_rsa'

clients_hosts = ParallelSSHClient(hosts, port=port, pkey=pkey, user=user, password=password)
seperate_clients_hosts = {}
for host in hosts:
    seperate_clients_hosts[host] = ParallelSSHClient([host], pkey=pkey, user=user, password=password)

def kill_all():
    for host in seperate_clients_hosts:
        print(host)
        output = host.run_command('ps aux | grep torchrun | grep -v grep | awk "{print \$2}" | sudo xargs kill -9 ', sudo=True)
        for host_out in output:
            host_out.stdin.write(password + '\n')
            host_out.stdin.flush()
        host.join(output)
        for line in host_out.stdout:
            print(line)
        for line in host_out.stderr:
            print(line)

def profile_local_p2p():
    output = seperate_clients_hosts[localhost_ip].run_command(f'cd {project_dir}/profiler; python profile_local_p2p.py')
    for host in output:
        host.join(output[host])
        for line in output[host].stdout:
            print(line)
        for line in output[host].stderr:
            print(line)

def profile_dist_p2p():
    res = []
    for idx, host in enumerate(hosts):
        output = seperate_clients_hosts[host].run_command(f'cd {project_dir}/profiler; python profile_dist_p2p.py {idx}')
        res.append(output)
    for host_out in res:
        for host in host_out:
            host.join(host_out[host])
            for line in host_out[host].stdout:
                print(line)
            for line in host_out[host].stderr:
                print(line)

