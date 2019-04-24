#!/usr/bin/env python
# forked from launch_8gpu.py

import argparse
import ncluster

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='megatron16',
                    help="name of the current run, used for machine naming and tensorboard visualization")
parser.add_argument('--machines', type=int, default=2,
                    help="how many machines to use")
parser.add_argument('--instance_type', type=str, default="p3dn.24xlarge",
                    help="which instance type to use")
parser.add_argument('--image_name', type=str,
                    default='Deep Learning AMI (Ubuntu) Version 22.0',
                    help="name of AMI to use ")
parser.add_argument('--nccl_rings', action='store_true', default=False,
                    help='use special nccl ring setup')
args = parser.parse_args()

ncluster.set_backend('aws')


def get_nccl_rings(num_tasks, num_gpus):
    ring = build_ring_order(range(num_tasks), range(num_gpus))
    ring_rev = build_ring_order(reversed(range(num_tasks)),
                                reversed(range(num_gpus)))
    rotated_gpu_order = [3, 2, 1, 0, 7, 6, 5, 4]
    skip_gpu_order = get_skip_order(num_gpus)
    if (num_tasks >= 4) and (num_gpus == 8):
        assert ((num_tasks % 4) == 0)
        skip_machine_order = get_skip_order(num_tasks)
        ring_skip = build_ring_order(skip_machine_order, rotated_gpu_order)
        ring_skip_rev = build_ring_order(reversed(skip_machine_order),
                                         skip_gpu_order)
        rings_arr = [ring, ring_rev, ring_skip, ring_skip_rev]
        # rings_arr = [ring, ring_rev, ring_skip]
    else:
        rings_arr = [ring, ring_rev]
    return ' | '.join(rings_arr)


def build_ring_order(machine_order, gpu_order):
    gpu_order = list(gpu_order)
    machine_order = list(machine_order)
    ngpus = len(gpu_order)
    r_order = [(x * ngpus) + y for x in machine_order for y in gpu_order]
    return ' '.join(map(str, r_order))


def get_skip_order(size):
    if size == 4:
        return [0, 2, 1, 3]
    skip_step = 5 if size == 16 else 3
    # step size of 3 yields - [0,3,6,1,4,7,2,5]
    return [(i * skip_step) % size for i in range(size)]


# routines to build NCCL ring orders
def get_nccl_params(num_tasks, num_gpus):
    if num_tasks <= 1 or not args.nccl_rings:
        return 'NCCL_DEBUG=VERSION'
    nccl_rings = get_nccl_rings(num_tasks, num_gpus)
    return f'NCCL_RINGS="{nccl_rings}" NCCL_SINGLE_RING_THRESHOLD=10 NCCL_DEBUG=VERSION'


def main():
    job = ncluster.make_job(name=args.name,
                            run_name=f"{args.name}",
                            num_tasks=args.machines,
                            image_name=args.image_name,
                            instance_type=args.instance_type)

    job.upload('*')
    job.run('killall python || echo failed')  # kill previous run
    job.run('source activate pytorch_p36')
    job.run('export NCCL_SOCKET_IFNAME=ens5')  # tip from cakarak@amazon.com
    job.run('pip install -r requirements.txt')
    # workaround for https://github.com/tensorflow/models/issues/3995
    job.run('pip install -U protobuf')

    num_gpus = 8
    assert args.instance_type in ['p3.16xlarge', 'p3dn.24xlarge'], f"{args.instance_type} is not 8-gpu"

    # WORLD_SIZE = num_gpus * args.machines
    MASTER_ADDR = job.tasks[0].ip
    MASTER_PORT = 6016
    NNODES = args.machines

    train = open('bookcorpus.filelist.train').read()
    validate = "/ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000163"
    test = "/ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000164"

    nccl_params = get_nccl_params(args.machines, num_gpus)

    lr = 0.0001   # original learning rate for 256 global batch size

    for i, task in enumerate(job.tasks):
        NODE_RANK = i
        DISTRIBUTED_ARGS = f"--nproc_per_node {num_gpus} --nnodes {NNODES} --node_rank {NODE_RANK} --master_addr {MASTER_ADDR} --master_port {MASTER_PORT}"

        cmd = (f"{nccl_params} python -m torch.distributed.launch {DISTRIBUTED_ARGS} "
               f"pretrain_bert.py "
               f"--batch-size 4 "
               f"--tokenizer-type BertWordPieceTokenizer "
               f"--cache-dir cache_dir "
               f"--tokenizer-model-type bert-large-uncased "
               f"--vocab-size 30522 "
               f"--use-tfrecords "
               f"--train-data {train} "
               f"--valid-data {validate} "
               f"--test-data {test} "
               f"--max-preds-per-seq 80 "
               f"--seq-length 512 "
               f"--max-position-embeddings 512 "
               f"--num-layers 24 "
               f"--hidden-size 1024 "
               f"--intermediate-size 4096 "
               f"--num-attention-heads 16 "
               f"--hidden-dropout 0.1 "
               f"--attention-dropout 0.1 "
               f"--train-iters 1000000 "
               f"--lr {lr} "
               f"--lr-decay-style linear "
               f"--lr-decay-iters 990000 "
               f"--warmup .01 "
               f"--weight-decay 1e-2 "
               f"--clip-grad 1.0 "
               f"--fp16 "
               f"--fp32-layernorm "
               f"--fp32-embedding "
               f"--hysteresis 2 "
               f"--num-workers 2 ")

        # new params
        cmd += f"--logdir {job.logdir} "

        task.run(f'echo {cmd} > {job.logdir}/task-{i}.cmd')  # save command-line
        task.run(cmd, non_blocking=True)

    print(f"Logging to {job.logdir}")


if __name__ == '__main__':
    main()
