import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import ray
import os
import numpy as np
import random
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None

from model import PolicyNet, QNet
from runner import RLRunner
from parameter import *

ray.init()
print("Welcome to RL autonomous exploration!")

writer = SummaryWriter(train_path)
if not os.path.exists(model_path):
    os.makedirs(model_path)
if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)

TENSORBOARD_METRIC_TAGS = [
    'Perf/Reward',
    'Losses/Value',
    'Losses/Policy Loss',
    'Losses/Q Value Loss',
    'Losses/Entropy',
    'Losses/Policy Grad Norm',
    'Losses/Q Value Grad Norm',
    'Losses/Log Alpha',
    'Losses/Alpha Loss',
    'Perf/Travel Distance',
    'Perf/Success Rate',
    'Perf/Explored Rate',
    'Features/Selected Expected Unknown Gain',
    'Features/Selected Frontier Cluster Size',
    'Features/Selected Basin Utility Sum',
    'Features/Selected Basin Expected Unknown Gain Sum',
    'Features/Selected Basin Frontier Cluster Max',
    'Features/Selected Basin Unvisited Ratio',
    'Features/Selected Basin Min Dist To Utility',
]


def init_wandb(run_name):
    if not USE_WANDB:
        return None

    if wandb is None:
        print("wandb is not installed. Install it with `pip install wandb` to enable wandb logging.")
        return None

    config = {
        name: value
        for name, value in globals().items()
        if name.isupper() and isinstance(value, (int, float, str, bool, type(None)))
    }
    config.update({
        'model_path': model_path,
        'train_path': train_path,
        'gifs_path': gifs_path,
    })

    return wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        name=run_name,
        mode=WANDB_MODE,
        config=config,
        dir=train_path,
        sync_tensorboard=False,
    )


def writeToTensorBoard(writer, tensorboardData, curr_episode, wandb_run=None):
    # each row in tensorboardData represents an episode
    # each column is a specific metric

    tensorboardData = np.array(tensorboardData)
    tensorboardData = list(np.nanmean(tensorboardData, axis=0))
    metric_values = [float(value) for value in tensorboardData]
    metrics = dict(zip(TENSORBOARD_METRIC_TAGS, metric_values))

    for tag, value in metrics.items():
        writer.add_scalar(tag=tag, scalar_value=value, global_step=curr_episode)

    if wandb_run is not None:
        wandb_run.log(metrics, step=curr_episode)


def format_elapsed_for_name(start_time):
    total_seconds = int((datetime.now() - start_time).total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    return f"{hours:02d}h{minutes:02d}m{seconds:02d}s"


def finalize_model_run_dir(model_run_dir, training_start_time):
    elapsed_str = format_elapsed_for_name(training_start_time)
    base_dir = os.path.dirname(model_run_dir)
    run_prefix = os.path.basename(model_run_dir).split('__elapsed_')[0]
    target_dir = os.path.join(base_dir, f"{run_prefix}__elapsed_{elapsed_str}")

    if model_run_dir == target_dir:
        return model_run_dir

    if os.path.exists(target_dir):
        suffix = 1
        while os.path.exists(f"{target_dir}_{suffix}"):
            suffix += 1
        target_dir = f"{target_dir}_{suffix}"

    os.rename(model_run_dir, target_dir)
    return target_dir


def main():
    # use GPU/CPU for driver/worker
    device = torch.device('cuda') if USE_GPU_GLOBAL else torch.device('cpu')
    local_device = torch.device('cuda') if USE_GPU else torch.device('cpu')
    
    # initialize neural networks
    global_policy_net = PolicyNet(INPUT_DIM, EMBEDDING_DIM, ACTION_FEATURE_DIM).to(device)
    global_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM, ACTION_FEATURE_DIM, USE_BASIN_IN_Q).to(device)
    global_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM, ACTION_FEATURE_DIM, USE_BASIN_IN_Q).to(device)
    log_alpha = torch.FloatTensor([-2]).to(device)  # not trainable when loaded from checkpoint, manually tune it for now
    log_alpha.requires_grad = True

    global_target_q_net1 = QNet(INPUT_DIM, EMBEDDING_DIM, ACTION_FEATURE_DIM, USE_BASIN_IN_Q).to(device)
    global_target_q_net2 = QNet(INPUT_DIM, EMBEDDING_DIM, ACTION_FEATURE_DIM, USE_BASIN_IN_Q).to(device)
    
    # initialize optimizers
    global_policy_optimizer = optim.Adam(global_policy_net.parameters(), lr=LR)
    global_q_net1_optimizer = optim.Adam(global_q_net1.parameters(), lr=LR)
    global_q_net2_optimizer = optim.Adam(global_q_net2.parameters(), lr=LR)
    log_alpha_optimizer = optim.Adam([log_alpha], lr=1e-4)

    # initialize decay (not use)
    policy_lr_decay = optim.lr_scheduler.StepLR(global_policy_optimizer, step_size=DECAY_STEP, gamma=0.96)
    q_net1_lr_decay = optim.lr_scheduler.StepLR(global_q_net1_optimizer,step_size=DECAY_STEP, gamma=0.96)
    q_net2_lr_decay = optim.lr_scheduler.StepLR(global_q_net2_optimizer,step_size=DECAY_STEP, gamma=0.96)
    log_alpha_lr_decay = optim.lr_scheduler.StepLR(log_alpha_optimizer, step_size=DECAY_STEP, gamma=0.96)
    
    # target entropy for SAC
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))

    training_start_time = datetime.now()
    print(f"Training started at {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    run_start_str = training_start_time.strftime('%Y%m%d_%H%M%S')
    run_name = f"{FOLDER_NAME}_{run_start_str}"
    wandb_run = init_wandb(run_name)
    model_run_dir = os.path.join(model_path, f"run_{run_start_str}__elapsed_running")
    os.makedirs(model_run_dir, exist_ok=True)
    print(f"Model checkpoints directory: {model_run_dir}")

    curr_episode = 0
    target_q_update_counter = 1

    # load model and optimizer trained before
    if LOAD_MODEL:
        print('Loading Model...')
        checkpoint = torch.load(model_path + '/checkpoint.pth')
        global_policy_net.load_state_dict(checkpoint['policy_model'])
        global_q_net1.load_state_dict(checkpoint['q_net1_model'])
        global_q_net2.load_state_dict(checkpoint['q_net2_model'])
        # log_alpha = checkpoint['log_alpha']  # not trainable when loaded from checkpoint, manually tune it for now
        global_policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        global_q_net1_optimizer.load_state_dict(checkpoint['q_net1_optimizer'])
        global_q_net2_optimizer.load_state_dict(checkpoint['q_net2_optimizer'])
        log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer'])
        policy_lr_decay.load_state_dict(checkpoint['policy_lr_decay'])
        q_net1_lr_decay.load_state_dict(checkpoint['q_net1_lr_decay'])
        q_net2_lr_decay.load_state_dict(checkpoint['q_net2_lr_decay'])
        log_alpha_lr_decay.load_state_dict(checkpoint['log_alpha_lr_decay'])
        curr_episode = checkpoint['episode']

        print("curr_episode set to ", curr_episode)
        print(log_alpha)
        print(global_policy_optimizer.state_dict()['param_groups'][0]['lr'])

    global_target_q_net1.load_state_dict(global_q_net1.state_dict())
    global_target_q_net2.load_state_dict(global_q_net2.state_dict())
    global_target_q_net1.eval()
    global_target_q_net2.eval()

    # launch meta agents
    meta_agents = [RLRunner.remote(i) for i in range(NUM_META_AGENT)]

    # get global networks weights
    weights_set = []
    if device != local_device:
        policy_weights = global_policy_net.to(local_device).state_dict()
        q_net1_weights = global_q_net1.to(local_device).state_dict()
        global_policy_net.to(device)
        global_q_net1.to(device)
    else:
        policy_weights = global_policy_net.to(local_device).state_dict()
        q_net1_weights = global_q_net1.to(local_device).state_dict()
    weights_set.append(policy_weights)
    weights_set.append(q_net1_weights)

    # distributed training if multiple GPUs available
    dp_policy = nn.DataParallel(global_policy_net)
    dp_q_net1 = nn.DataParallel(global_q_net1)
    dp_q_net2 = nn.DataParallel(global_q_net2)
    dp_target_q_net1 = nn.DataParallel(global_target_q_net1)
    dp_target_q_net2 = nn.DataParallel(global_target_q_net2)

    # launch the first job on each runner
    job_list = []
    for i, meta_agent in enumerate(meta_agents):
        curr_episode += 1
        job_list.append(meta_agent.job.remote(weights_set, curr_episode))
    
    # initialize metric collector
    metric_name = [
        'travel_dist',
        'success_rate',
        'explored_rate',
        'selected_expected_unknown_gain',
        'selected_frontier_cluster_size',
        'selected_basin_utility_sum',
        'selected_basin_expected_unknown_gain_sum',
        'selected_basin_frontier_cluster_max',
        'selected_basin_unvisited_ratio',
        'selected_basin_min_dist_to_utility',
    ]
    training_data = []
    perf_metrics = {}
    for n in metric_name:
        perf_metrics[n] = []

    # initialize training replay buffer
    experience_buffer = []
    for i in range(17):
        experience_buffer.append([])
    
    # collect data from worker and do training
    try:
        while True:
            # wait for any job to be completed
            done_id, job_list = ray.wait(job_list)
            # get the results
            done_jobs = ray.get(done_id)
            
            # save experience and metric
            for job in done_jobs:
                job_results, metrics, info = job
                for i in range(len(experience_buffer)):
                    experience_buffer[i] += job_results[i]
                for n in metric_name:
                    perf_metrics[n].append(metrics[n])

            # launch new task
            curr_episode += 1
            job_list.append(meta_agents[info['id']].job.remote(weights_set, curr_episode))
            
            # start training
            if curr_episode % 1 == 0 and len(experience_buffer[0]) >= MINIMUM_BUFFER_SIZE:
                print("training")

                # keep the replay buffer size
                if len(experience_buffer[0]) >= REPLAY_SIZE:
                    for i in range(len(experience_buffer)):
                        experience_buffer[i] = experience_buffer[i][-REPLAY_SIZE:]

                indices = range(len(experience_buffer[0]))

                # training for n times each step
                for j in range(8):
                    # randomly sample a batch data
                    sample_indices = random.sample(indices, BATCH_SIZE)
                    rollouts = []
                    for i in range(len(experience_buffer)):
                        rollouts.append([experience_buffer[i][index] for index in sample_indices])

                    # stack batch data to tensors
                    node_inputs_batch = torch.stack(rollouts[0]).to(device)
                    edge_inputs_batch = torch.stack(rollouts[1]).to(device)
                    current_inputs_batch = torch.stack(rollouts[2]).to(device)
                    node_padding_mask_batch = torch.stack(rollouts[3]).to(device)
                    edge_padding_mask_batch = torch.stack(rollouts[4]).to(device)
                    edge_mask_batch = torch.stack(rollouts[5]).to(device)
                    action_features_batch = torch.stack(rollouts[6]).to(device)
                    action_batch = torch.stack(rollouts[7]).to(device)
                    reward_batch = torch.stack(rollouts[8]).to(device)
                    done_batch = torch.stack(rollouts[9]).to(device)
                    next_node_inputs_batch = torch.stack(rollouts[10]).to(device)
                    next_edge_inputs_batch = torch.stack(rollouts[11]).to(device)
                    next_current_inputs_batch = torch.stack(rollouts[12]).to(device)
                    next_node_padding_mask_batch = torch.stack(rollouts[13]).to(device)
                    next_edge_padding_mask_batch = torch.stack(rollouts[14]).to(device)
                    next_edge_mask_batch = torch.stack(rollouts[15]).to(device)
                    next_action_features_batch = torch.stack(rollouts[16]).to(device)

                    # SAC
                    with torch.no_grad():
                        q_values1, _ = dp_q_net1(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch, action_features_batch)
                        q_values2, _ = dp_q_net2(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch, action_features_batch)
                        q_values = torch.min(q_values1, q_values2)

                    logp = dp_policy(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch, action_features_batch)
                    policy_loss = torch.sum((logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_values.detach())), dim=1).mean()

                    global_policy_optimizer.zero_grad()
                    policy_loss.backward()
                    policy_grad_norm = torch.nn.utils.clip_grad_norm_(global_policy_net.parameters(), max_norm=100, norm_type=2)
                    global_policy_optimizer.step()
                    
                    with torch.no_grad():
                        next_logp = dp_policy(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch, next_action_features_batch)
                        next_q_values1, _ = dp_target_q_net1(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch, next_action_features_batch)
                        next_q_values2, _ = dp_target_q_net2(next_node_inputs_batch, next_edge_inputs_batch, next_current_inputs_batch, next_node_padding_mask_batch, next_edge_padding_mask_batch, next_edge_mask_batch, next_action_features_batch)
                        next_q_values = torch.min(next_q_values1, next_q_values2)
                        value_prime_batch = torch.sum(next_logp.unsqueeze(2).exp() * (next_q_values - log_alpha.exp() * next_logp.unsqueeze(2)), dim=1).unsqueeze(1)
                        target_q_batch = reward_batch + GAMMA * (1 - done_batch) * value_prime_batch
                    
                    mse_loss = nn.MSELoss()
                    q_values1, _ = dp_q_net1(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch, action_features_batch)
                    q1 = torch.gather(q_values1, 1, action_batch)
                    q1_loss = mse_loss(q1, target_q_batch.detach()).mean()

                    global_q_net1_optimizer.zero_grad()
                    q1_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net1.parameters(), max_norm=20000, norm_type=2)
                    global_q_net1_optimizer.step()
                    
                    q_values2, _ = dp_q_net2(node_inputs_batch, edge_inputs_batch, current_inputs_batch, node_padding_mask_batch, edge_padding_mask_batch, edge_mask_batch, action_features_batch)
                    q2 = torch.gather(q_values2, 1, action_batch)
                    q2_loss = mse_loss(q2, target_q_batch.detach()).mean()

                    global_q_net2_optimizer.zero_grad()
                    q2_loss.backward()
                    q_grad_norm = torch.nn.utils.clip_grad_norm_(global_q_net2.parameters(), max_norm=20000, norm_type=2)
                    global_q_net2_optimizer.step()

                    entropy = (logp * logp.exp()).sum(dim=-1)
                    alpha_loss = -(log_alpha * (entropy.detach() + entropy_target)).mean()

                    log_alpha_optimizer.zero_grad()
                    alpha_loss.backward()
                    log_alpha_optimizer.step()

                    target_q_update_counter += 1
                    #print("target q update counter", target_q_update_counter % 1024)

                #policy_lr_decay.step()
                #q_net1_lr_decay.step()
                #q_net2_lr_decay.step()
                #log_alpha_lr_decay.step()

                # data record to be written in tensorboard
                perf_data = []
                for n in metric_name:
                    perf_data.append(np.nanmean(perf_metrics[n]))
                data = [reward_batch.mean().item(), value_prime_batch.mean().item(), policy_loss.item(), q1_loss.item(),
                        entropy.mean().item(), policy_grad_norm.item(), q_grad_norm.item(), log_alpha.item(), alpha_loss.item(), *perf_data]
                training_data.append(data)

            # write record to tensorboard
            if len(training_data) >= SUMMARY_WINDOW:
                elapsed_time = datetime.now() - training_start_time
                elapsed_str = str(elapsed_time).split('.')[0]
                print(f"[Episode {curr_episode}] Elapsed training time: {elapsed_str}")
                writeToTensorBoard(writer, training_data, curr_episode, wandb_run)
                training_data = []
                perf_metrics = {}
                for n in metric_name:
                    perf_metrics[n] = []

            # get the updated global weights
            weights_set = []
            if device != local_device:
                policy_weights = global_policy_net.to(local_device).state_dict()
                q_net1_weights = global_q_net1.to(local_device).state_dict()
                global_policy_net.to(device)
                global_q_net1.to(device)
            else:
                policy_weights = global_policy_net.to(local_device).state_dict()
                q_net1_weights = global_q_net1.to(local_device).state_dict()
            weights_set.append(policy_weights)
            weights_set.append(q_net1_weights)
            
            # update the target q net
            if target_q_update_counter > 64:
                print("update target q net")
                target_q_update_counter = 1
                global_target_q_net1.load_state_dict(global_q_net1.state_dict())
                global_target_q_net2.load_state_dict(global_q_net2.state_dict())
                global_target_q_net1.eval()
                global_target_q_net2.eval()

            # save the model
            should_save_checkpoint = curr_episode % 32 == 0
            should_sparse_archive_checkpoint = (
                curr_episode >= ARCHIVE_CHECKPOINT_START_EPISODE and
                (curr_episode - ARCHIVE_CHECKPOINT_START_EPISODE) % ARCHIVE_CHECKPOINT_GAP == 0
            )
            should_frequent_archive_checkpoint = (
                curr_episode >= ARCHIVE_CHECKPOINT_FREQUENT_START_EPISODE and
                (curr_episode - ARCHIVE_CHECKPOINT_FREQUENT_START_EPISODE) % ARCHIVE_CHECKPOINT_FREQUENT_GAP == 0
            )
            should_archive_checkpoint = should_sparse_archive_checkpoint or should_frequent_archive_checkpoint
            if should_save_checkpoint or should_archive_checkpoint:
                print('Saving model', end='\n')
                checkpoint = {"policy_model": global_policy_net.state_dict(),
                                "q_net1_model": global_q_net1.state_dict(),
                                "q_net2_model": global_q_net2.state_dict(),
                                "log_alpha": log_alpha,
                                "policy_optimizer": global_policy_optimizer.state_dict(),
                                "q_net1_optimizer": global_q_net1_optimizer.state_dict(),
                                "q_net2_optimizer": global_q_net2_optimizer.state_dict(),
                                "log_alpha_optimizer": log_alpha_optimizer.state_dict(),
                                "episode": curr_episode,
                                "policy_lr_decay": policy_lr_decay.state_dict(),
                                "q_net1_lr_decay": q_net1_lr_decay.state_dict(),
                                "q_net2_lr_decay": q_net2_lr_decay.state_dict(),
                                "log_alpha_lr_decay": log_alpha_lr_decay.state_dict()
                        }

                if should_save_checkpoint:
                    path_checkpoint = os.path.join(model_run_dir, 'checkpoint.pth')
                    torch.save(checkpoint, path_checkpoint)
                    print(f'Saved model to {path_checkpoint}', end='\n')

                if should_archive_checkpoint:
                    archive_checkpoint = os.path.join(model_run_dir, f'checkpoint_episode_{curr_episode}.pth')
                    torch.save(checkpoint, archive_checkpoint)
                    print(f'Saved archive checkpoint to {archive_checkpoint}', end='\n')
                    
    
    except KeyboardInterrupt:
        print("CTRL_C pressed. Killing remote workers")
        for a in meta_agents:
            ray.kill(a)
    finally:
        if os.path.exists(model_run_dir):
            final_model_run_dir = finalize_model_run_dir(model_run_dir, training_start_time)
            print(f"Final checkpoint directory: {final_model_run_dir}")
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
