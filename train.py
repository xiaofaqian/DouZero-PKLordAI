import os

from douzero.dmc import parser, train

if __name__ == '__main__':
    flags = parser.parse_args()
    flags.actor_device_cpu = True
    flags.training_device = 'cpu'
    flags.save_interval = 5
    flags.load_model = True
    os.environ["CUDA_VISIBLE_DEVICES"] = flags.gpu_devices
    train(flags)
