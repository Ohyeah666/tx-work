import ray
import numpy as np
import os
import torch
from datetime import datetime

from model import PolicyNet
from test_worker import TestWorker
from test_parameter import *


def format_elapsed_for_name(start_time):
    total_seconds = int((datetime.now() - start_time).total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


def make_unique_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return path

    suffix = 1
    while os.path.exists(f"{path}_{suffix}"):
        suffix += 1

    unique_path = f"{path}_{suffix}"
    os.makedirs(unique_path)
    return unique_path


def create_test_gifs_dir(test_set_name, test_start_time):
    run_start_str = test_start_time.strftime('%Y%m%d_%H%M%S')
    test_run_dir = os.path.join(gifs_path, f"{test_set_name}_{run_start_str}__elapsed_running")
    return make_unique_dir(test_run_dir)


def finalize_test_gifs_dir(test_run_dir, test_start_time):
    elapsed_str = format_elapsed_for_name(test_start_time)
    base_dir = os.path.dirname(test_run_dir)
    run_prefix = os.path.basename(test_run_dir).split('__elapsed_')[0]
    target_dir = os.path.join(base_dir, f"{run_prefix}__elapsed_{elapsed_str}")

    if test_run_dir == target_dir:
        return test_run_dir

    if os.path.exists(target_dir):
        suffix = 1
        while os.path.exists(f"{target_dir}_{suffix}"):
            suffix += 1
        target_dir = f"{target_dir}_{suffix}"

    os.rename(test_run_dir, target_dir)
    return target_dir


def run_test():
    if not os.path.exists(trajectory_path):
        os.makedirs(trajectory_path)

    test_start_time = datetime.now()
    test_gifs_path = None
    if SAVE_GIFS:
        test_gifs_path = create_test_gifs_dir(TEST_SET_NAME, test_start_time)
        print(f"Test GIFs directory: {test_gifs_path}")

    device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    global_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM).to(device)

    if device == 'cuda':
        checkpoint = torch.load(f'{model_path}/checkpoint_episode_17600.pth')
    else:
        checkpoint = torch.load(f'{model_path}/checkpoint_episode_17600.pth', map_location = torch.device('cpu'))

    global_network.load_state_dict(checkpoint['policy_model'])

    meta_agents = [Runner.remote(i, TEST_SET_NAME, test_gifs_path) for i in range(NUM_META_AGENT)]
    weights = global_network.state_dict()
    curr_test = 0

    dist_history = []

    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        job_list.append(meta_agent.job.remote(weights, curr_test))
        curr_test += 1

    try:
        while len(dist_history) < curr_test:
            done_id, job_list = ray.wait(job_list)
            done_jobs = ray.get(done_id)

            for job in done_jobs:
                metrics, info = job
                dist_history.append(metrics['travel_dist'])
            if curr_test < NUM_TEST:
                job_list.append(meta_agents[info['id']].job.remote(weights, curr_test))
                curr_test += 1

        print('|#Total test:', NUM_TEST)
        print('|#Average length:', np.array(dist_history).mean())
        print('|#Length std:', np.array(dist_history).std())

    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
    finally:
        if SAVE_GIFS and test_gifs_path and os.path.exists(test_gifs_path):
            final_gifs_path = finalize_test_gifs_dir(test_gifs_path, test_start_time)
            print(f"Final test GIFs directory: {final_gifs_path}")


@ray.remote(num_cpus=1, num_gpus=NUM_GPU/NUM_META_AGENT)
class Runner(object):
    def __init__(self, meta_agent_id, test_set_name, gifs_dir):
        self.meta_agent_id = meta_agent_id
        self.test_set_name = test_set_name
        self.gifs_dir = gifs_dir
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def set_weights(self, weights):
        self.local_network.load_state_dict(weights)

    def do_job(self, episode_number):
        worker = TestWorker(self.meta_agent_id, self.local_network, episode_number, device=self.device,
                            save_image=SAVE_GIFS, greedy=True, gifs_dir=self.gifs_dir,
                            test_set_name=self.test_set_name)
        worker.work(episode_number)

        perf_metrics = worker.perf_metrics
        return perf_metrics

    def job(self, weights, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        # set the local weights to the global weight values from the master network
        self.set_weights(weights)

        metrics = self.do_job(episode_number)

        info = {
            "id": self.meta_agent_id,
            "episode_number": episode_number,
        }

        return metrics, info


if __name__ == '__main__':
    ray.init()
    for i in range(NUM_RUN):
        run_test()
