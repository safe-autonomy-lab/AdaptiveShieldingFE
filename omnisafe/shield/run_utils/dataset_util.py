import numpy as np
import jax.numpy as jnp
from ..util import derivative_of

def get_batches(states_actions, next_states, batch_size):
    dataset_size = len(states_actions)
    indices = np.random.permutation(dataset_size)
    for i in range(0, dataset_size, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield (states_actions[batch_indices], 
            next_states[batch_indices])

def process_transitions_safety_gym(transitions, max_hidden: int = 1000):
    hiddens = list(transitions.keys())
    # hiddens = sorted(hiddens)
    train_in = []
    train_target = []
    num_samples = np.inf
    hiddens_to_remove = []
    for hidden in hiddens[:max_hidden]:
        train_state_action = []
        train_next_position = []
        for (x, y) in transitions[hidden]:
            train_state_action.append(x)
            train_next_position.append(y)
            
        if np.shape(train_state_action)[0] < 250:
            hiddens_to_remove.append(hidden)
        else:
            num_samples = min(np.shape(train_state_action)[0], num_samples)
            
    for hidden in hiddens_to_remove:
        hiddens.remove(hidden)
    
    for hidden in hiddens[:max_hidden]:
        train_state_action = []
        train_next_position = []
        for (x, y) in transitions[hidden]:
            train_state_action.append(x)
            train_next_position.append(y)
        
        train_in.append(jnp.array(train_state_action[:num_samples])[None])
        train_target.append(jnp.array(train_next_position[:num_samples])[None])
    
    return jnp.concatenate(train_in, axis=0), jnp.concatenate(train_target, axis=0), list(hiddens)

def create_mo_training_dataset(ped_episode_transitions, history_length=5, future_length=1, add_extra=True):
    X = []
    Y = []
    for epi in ped_episode_transitions:
        ped_trans = np.array(ped_episode_transitions[epi])
        # steps, nbr_of_peds, feature_dim = ped_trans.shape
        ped_trans = np.transpose(ped_trans, (1, 0, 2))
        num_pedestrians, time_steps, feature_dim = ped_trans.shape

        # Create X by stacking history for each time step
        if time_steps <= history_length or time_steps - history_length < future_length:
            continue

        X_stacked = []
        for t in range(history_length, time_steps - future_length):
            # Get position history and compute derivatives
            history = ped_trans[:, t-history_length:t, :2].copy()  # [num_ped, history_len, 2]
            if add_extra:
                vx = derivative_of(history[:, :, 0], dt=0.02)
                vy = derivative_of(history[:, :, 1], dt=0.02)
                history = np.stack([history[..., 0], history[..., 1], vx, vy], axis=-1)  # [num_ped, history_len, 6]
            
            X_stacked.append(history)
        
        stacked_history = np.array(X_stacked).transpose(1, 0, 2, 3)
        X.append(stacked_history)
        
        # Create Y by taking the next position (indices 1 and 2) for the corresponding time steps
        Y_stacked = []
        for t in range(history_length, time_steps - future_length):
            future_positions = ped_trans[:, t: t + future_length, :2]
            Y_stacked.append(future_positions)

        stacked_future_positions = np.array(Y_stacked).transpose(1, 0, 2, 3)
        first_dim, sec_dim, third_dim, fourth_dim = stacked_future_positions.shape
        Y.append(stacked_future_positions.reshape(first_dim, sec_dim, -1))
    
    return np.stack(X), np.stack(Y)
