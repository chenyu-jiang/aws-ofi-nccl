import os
import shutil
import subprocess
import argparse
from collections import defaultdict

def add_parser_args(parser):
    # adds common parser args, returns nothing
    parser.add_argument("-c", "--config", help="mcconfig file (one rank execs one session in the file)", type=str, required=True)
    parser.add_argument("-n", "--n-nodes", help="number of nodes to run on", type=int, required=True)
    parser.add_argument("-g", "--n-gpus-per-node", help="number of gpus per node, default 8", type=int, default=8)
    parser.add_argument(
        "-r", "--requests", help="number of requests per iteration, default 128", type=int, default=128
    )
    parser.add_argument(
        "-i",
        "--iters",
        help="Number of iterations to run, default 100",
        type=int,
        default=100
    )
    parser.add_argument(
        "-w",
        "--warmup-iters",
        help="Number of warmup iterations to run, default 20",
        type=int,
        default=20
    )
    parser.add_argument(
        "-s",
        "--size",
        help="Size of each send request, default 1MB.",
        type=int,
        default=1048576
    )

def gen_send_args(args):
    # args: args from parser
    # returns: list of args

    def get_node(dev_idx):
        return dev_idx // args.n_gpus_per_node

    mc_sessions = []

    with open(args.config, "r") as f:
        for idx, line in enumerate(f):
            if line.startswith("#"):
                continue
            line = line.strip()
            if not line:
                continue
            splitted_line = line.split()
            assert len(splitted_line) == 3, f"Invalid line {idx} in mcconfig: {line}"
            src_device, dst_device, _ = splitted_line
            src_device = int(src_device)
            dst_device = int(dst_device)
            assert (
                0 <= src_device < args.n_nodes * args.n_gpus_per_node
            ), f"Invalid source device {src_device} in line {idx} in mcconfig"
            assert (
                0 <= dst_device < args.n_nodes * args.n_gpus_per_node
            ), f"Invalid destination device {dst_device} in line {idx} in mcconfig"
            assert get_node(src_device) != get_node(
                dst_device
            ), f"Invalid device pair ({src_device},{dst_device}) in line {idx} in mcconfig. Src and dst device must locate on different nodes."
            mc_sessions.append((src_device, dst_device))

    def get_params(local_dev_id, remote_rank, is_client):
        return [str(args.requests),
                str(args.iters),
                str(args.warmup_iters),
                str(args.size),
                str(local_dev_id),
                str(remote_rank),
                str(int(is_client)),
            ]

    params = []
    node_send_sess_indices = defaultdict(list)
    node_recv_sess_indices = defaultdict(list)

    for session_idx, (src_idx, dst_idx) in enumerate(mc_sessions):
        src_node = get_node(src_idx)
        dst_node = get_node(dst_idx)
        node_send_sess_indices[src_node].append(session_idx)
        node_recv_sess_indices[dst_node].append(session_idx)

    curr_rank = 0
    rank2sess = {}
    srcsess2rank = {}
    dstsess2rank = {}
    for node_idx in range(args.n_nodes):
        for sess_idx in node_send_sess_indices[node_idx]:
            srcsess2rank[sess_idx] = curr_rank
            rank2sess[curr_rank] = (sess_idx, True)
            curr_rank += 1
        for sess_idx in node_recv_sess_indices[node_idx]:
            dstsess2rank[sess_idx] = curr_rank
            rank2sess[curr_rank] = (sess_idx, False)
            curr_rank += 1

    for rank in range(curr_rank):
        sess_idx, is_client = rank2sess[rank]
        src_idx, dst_idx = mc_sessions[sess_idx]

        if is_client:
            node = get_node(src_idx)
            local_dev = src_idx % args.n_gpus_per_node
            remote_rank = dstsess2rank[sess_idx]
        else:
            node = get_node(dst_idx)
            local_dev = dst_idx % args.n_gpus_per_node
            remote_rank = srcsess2rank[sess_idx]

        assert rank // ((curr_rank) // args.n_nodes) == node, f"Invalid node {node} for rank {rank}."

        params += get_params(local_dev, remote_rank, is_client)
        params.append("")

    return params, curr_rank

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate args for nccl_message_transfer tests according to mcconfig."
    )
    add_parser_args(parser)

    parser.add_argument(
        "-o",
        "--output",
        help="file to store the generated args, defaults to nccl_msg_<the name of mcconfig>.args",
        type=str,
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = "./nccl_msg_" + args.config.split(".")[0] + ".args"

    output_args, n_ranks = gen_send_args(args)

    print(f"Created args file with {n_ranks} total ranks.")

    with open(args.output, "w") as f:
        f.write("\n".join(output_args))
        f.write("\n") # we need an extra newline since join will not add \n to last element