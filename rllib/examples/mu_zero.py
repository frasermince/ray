from ray.rllib.algorithms.mu_zero import MuZeroConfig
config = MuZeroConfig().training(sgd_minibatch_size=256)\
 .resources(num_gpus=0).rollouts()
print(config.to_dict())
# Build a Algorithm object from the config and run 1 training iteration.
trainer = config.build(env="Pong-v4")
trainer.train()
