import time
from typing import Dict
from pssh.clients.ssh import SSHClient
import pprint
import math
import signal
import sys
import itertools
import json


def execute_command(clients: Dict[str, SSHClient], commands: Dict[str, str]):
    output = {}
    for host, command in commands.items():
        output[host] = (clients[host].run_command(command))
        print(f"executed command on {host}: {command}") 
        
    for host, client in clients.items():
        client.wait_finished(output[host])
    
    # for host, out in output.items():
    #     print(f"output of {host}:")
    #     for line in out.stdout:
    #         print(line)
    #     for line in out.stderr:
    #         print(line)

def kill_all(clients):
    for host in clients:
        print(host)
        output = clients[host].run_command(
            'ps aux | grep pretrain | grep -v grep | awk "{print \$2}" | sudo xargs kill -9 ',
            sudo=True,
        )
        host.join(output)
        for line in output.stdout:
            print(line)
        for line in output.stderr:
            print(line)

def setup_signal_handler(clients):
    def signal_handler(signum, frame):
        kill_all(clients)
        sys.exit(0) 
    return signal_handler




if __name__ == "__main__":

    # hosts = ["172.31.37.189", "172.31.33.155", "172.31.31.19", "172.31.25.221"]
    # hosts = ["172.31.37.189","172.31.33.155","172.31.31.19", "172.31.20.160"]
    hosts = ["172.31.37.189", "172.31.31.19"]
    container_project_dir = "/workspace/Hetaceso/runtime/tests/unit_tests/flexpipe/metis/"
    host_project_dir = "/home/ubuntu/code/python/Hetaceso/runtime/tests/unit_tests/flexpipe/metis"
    container_name = "hetaceso-ubuntu"
    pkey = "/home/ubuntu/.ssh/id_rsa"


    # init ssh clients
    clients = {}
    for host in hosts:
        clients[host] = SSHClient(host, pkey=pkey, user="ubuntu")
    
    signal.signal(signal.SIGINT, setup_signal_handler(clients))

    # model_names = ["GPT_1-3B", "GPT_2-6B", "GPT_6-7B"]
    # seq_lens = [8192, 16384, 32768, 65536]
    model_names = ["GPT_350M"]
    seq_lens = [4096]

    # generate commands for each experiment
    all_commands = {}

    for model_name, seq_len in itertools.product(model_names, seq_lens):
        
        gbs = 1024 // (seq_len // 8192)
        exp_key = f"{model_name}_{seq_len}"
        all_commands[exp_key] = {}
        host_flex_config = f"{host_project_dir}/config/{model_name}_seq-{seq_len}.json"
        container_flex_config = f"{container_project_dir}/config/{model_name}_seq-{seq_len}.json"
        # host_flex_config = f"{host_project_dir}/test.json"
        # container_flex_config = f"{container_project_dir}/test.json"
        with open(host_flex_config) as f:
            config = json.load(f)
            mbs = gbs // int(config["nums_of_mbs"])
            for node_rank, host in enumerate(hosts):
                all_commands[exp_key][
                    host
                ] = f'docker exec {container_name} bash -c "cd {container_project_dir} && ./run_rank.sh -n {node_rank} -m {model_name} -s {seq_len} -u {mbs} -g {gbs} -f {container_flex_config}"'

    pprint.pp(all_commands)

    # execute commands
    for model_name, seq_len in itertools.product(model_names, seq_lens):
        exp_key = f"{model_name}_{seq_len}"
        print(f"start: {exp_key}")
        execute_command(clients, all_commands[exp_key]) 
        print(f"end: {exp_key}")

