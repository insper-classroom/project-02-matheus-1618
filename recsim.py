import numpy as np
import tensorflow as tf
import recsim
# from recsim.environments import interest_evolution
# from recsim.agents import full_slate_q_agent
# from recsim.simulator import runner_lib


def create_agent(sess, environment, eval_mode, summary_writer=None):
  kwargs = {
      'observation_space': environment.observation_space,
      'action_space': environment.action_space,
      'summary_writer': summary_writer,
      'eval_mode': eval_mode,
  }
  return recsim.agents.full_slate_q_agent.FullSlateQAgent(sess, **kwargs)


seed = 0
np.random.seed(seed)
env_config = {
  'num_candidates': 10,
  'slate_size': 2,
  'resample_documents': True,
  'seed': seed,
  }


tmp_base_dir = '/tmp/recsim/'
runner = recsim.simulator.runner_lib.TrainRunner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_agent,
    env=recsim.environments.interest_evolution.create_environment(env_config),
    episode_log_file="",
    max_training_steps=50,
    num_iterations=10)
runner.run_experiment()


runner = recsim.simulator.runner_lib.EvalRunner(
    base_dir=tmp_base_dir,
    create_agent_fn=create_agent,
    env=recsim.environments.interest_evolution.create_environment(env_config),
    max_eval_episodes=5,
    test_mode=True)
runner.run_experiment()

