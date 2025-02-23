import time
from typing import Dict
from pssh.clients.ssh import ParallelSSHClient, SSHClient
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
    
    # for idx, out in enumerate(output):
    #     print(f"output of {idx}:")
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

    hosts = ["localhost", "10.156.154.242"]
    host_project_dirs = {
        "localhost": "/home/gzy/projects/Hetaceso",
        "10.156.154.242": "/home/gaoziyuan/projects/Hetaceso",
    }
    container_project_dir = "/workspace/Hetaceso/runtime/tests/unit_tests/flexpipe/metis/"
    contianer_names = {"localhost": "hetaceso-gzy", "10.156.154.242": "hetaceso-gaoziyuan"}
    users = {"localhost": "gzy", "10.156.154.242": "gaoziyuan"}
    password = "gaoziyuan"
    pkey = "/home/gzy/.ssh/id_rsa"


    # init ssh clients
    clients = {}
    for host in hosts:
        clients[host] = SSHClient(host, pkey=pkey, user="ubuntu")
    
    signal.signal(signal.SIGINT, setup_signal_handler(clients))

    model_names = ["GPT_1-3B", "GPT_2-6B", "GPT_6-7B"]
    seq_lens = [8192, 16384, 32768, 65536]
    gbs = 1024

    # generate commands for each experiment
    all_commands = {}

    for model_name, seq_len in itertools.product(model_names, seq_lens):
        exp_key = f"{model_name}_{seq_len}"
        all_commands[exp_key] = {}
        flex_config = f"{container_project_dir}/config/{model_name}_seq-{seq_len}.json"
        with open(flex_config) as f:
            config = json.load(f)
            mbs = gbs // int(json["nums_of_mbs"])
            for node_rank, host in enumerate(hosts):
                all_commands[exp_key][
                    host
                ] = f'docker exec {contianer_names[host]} bash -c "cd {container_project_dir} && ./run_rank.sh -n {node_rank} -m {model_name} -s {seq_len} -u {mbs} -g {gbs} -f {flex_config}"'

    pprint.pp(all_commands)

    # execute commands
    for model_name, seq_len in itertools.product(model_names, seq_lens):
        exp_key = f"{model_name}_{seq_len}"
        print(f"start: {exp_key}")
        execute_command(clients, all_commands[exp_key]) 
        print(f"end: {exp_key}")

