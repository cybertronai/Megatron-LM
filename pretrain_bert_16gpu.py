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

    train = "/ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0000 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0001 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0002 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0003 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0004 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0005 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0006 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0007 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0008 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord0009 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00010 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00011 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00012 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00013 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00014 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00015 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00016 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00017 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00018 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00019 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00020 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00021 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00022 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00023 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00024 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00025 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00026 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00027 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00028 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00029 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00030 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00031 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00032 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00033 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00034 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00035 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00036 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00037 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00038 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00039 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00040 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00041 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00042 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00043 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00044 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00045 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00046 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00047 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00048 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00049 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00050 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00051 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00052 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00053 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00054 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00055 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00056 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00057 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00058 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00059 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00060 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00061 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00062 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00063 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00064 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00065 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00066 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00067 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00068 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00069 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00070 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00071 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00072 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00073 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00074 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00075 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00076 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00077 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00078 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00079 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00080 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00081 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00082 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00083 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00084 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00085 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00086 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00087 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00088 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00089 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00090 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00091 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00092 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00093 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00094 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00095 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00096 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00097 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00098 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord00099 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000100 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000101 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000102 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000103 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000104 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000105 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000106 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000107 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000108 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000109 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000110 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000111 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000112 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000113 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000114 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000115 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000116 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000117 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000118 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000119 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000120 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000121 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000122 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000123 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000124 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000125 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000126 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000127 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000128 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000129 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000130 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000131 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000132 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000133 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000134 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000135 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000136 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000137 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000138 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000139 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000140 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000141 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000142 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000143 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000144 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000145 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000146 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000147 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000148 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000149 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000150 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000151 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000152 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000153 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000154 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000155 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000156 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000157 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000158 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000159 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000160 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000161 /ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000162"
    validate = "/ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000163"
    test = "/ncluster/data/bookcorpus.tfrecords/final_tfrecords_sharded/tf_examples.tfrecord000164"

    nccl_params = get_nccl_params(args.machines, num_gpus)

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
               f"--lr 0.0001 "
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
