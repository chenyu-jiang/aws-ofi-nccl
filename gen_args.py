import os
import shutil
import subprocess
import argparse

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

    for session_idx, (src_idx, dst_idx) in enumerate(mc_sessions):
        send_rank = session_idx * 2
        recv_rank = send_rank + 1
        src_node = get_node(src_idx)
        dst_node = get_node(dst_idx)

        send_local_dev = src_idx % args.n_gpus_per_node
        recv_local_dev = dst_idx % args.n_gpus_per_node

        params += get_params(send_local_dev, recv_rank, True)
        params.append("")

        params += get_params(recv_local_dev, send_rank, False)
        params.append("")

    return params, len(mc_sessions) * 2

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