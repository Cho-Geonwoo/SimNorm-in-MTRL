# Code backbone: Decision Transformer https://github.com/kzl/decision-transformer/
# Decision Transformer License: https://github.com/kzl/decision-transformer/blob/master/LICENSE.md

import numpy as np
import torch
import torch.nn as nn

import transformers

from .trajectory_gpt2 import GPT2Model

class PromptDecisionTransformer(nn.Module):

    def __init__(
            self,
            env_name_list,
            info,
            special_embedding,
            hidden_size,
            max_length=None,
            max_ep_len=4096,
            action_tanh=True,
            **kwargs
    ):
        super().__init__()
        self.env_name_list = env_name_list
        self.info = info
        self.special_embedding = special_embedding
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.max_ep_len = max_ep_len
        config = transformers.GPT2Config(vocab_size=1, n_embd=hidden_size, **kwargs)

        # note: the only difference between this GPT2Model and the default Huggingface version
        # is that the positional embeddings are removed (since we'll add those ourselves)
        self.transformer = GPT2Model(config)
        # change to parallelize mode for metaworld big model
        # self.transformer.parallelize()

        self.embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_ln = nn.LayerNorm(hidden_size)
        self.predict_return = torch.nn.Linear(hidden_size, 1)

        if special_embedding is True:
            for name in env_name_list:
                setattr(self, f"{name}_embed_state", torch.nn.Linear(self.info[name]['state_dim'], hidden_size))
                setattr(self, f"{name}_embed_action", torch.nn.Linear(self.info[name]['act_dim'], hidden_size))
                setattr(self, f"{name}_predict_state", torch.nn.Linear(hidden_size, self.info[name]['state_dim']))
                setattr(self, f"{name}_predict_action", nn.Sequential(
                    *([nn.Linear(hidden_size, self.info[name]['act_dim'])] + ([nn.Tanh()] if action_tanh else []))
                ))

                # prompt
                setattr(self, f"{name}_prompt_embed_state", torch.nn.Linear(self.info[name]['state_dim'], hidden_size))
                setattr(self, f"{name}_prompt_embed_action", torch.nn.Linear(self.info[name]['act_dim'], hidden_size))
                setattr(self, f"{name}_prompt_predict_action", torch.nn.Linear(self.info[name]['act_dim'], hidden_size))
        else:
            name = env_name_list[0]
            setattr(self, f"embed_state", torch.nn.Linear(self.info[name]['state_dim'], hidden_size))
            setattr(self, f"embed_action", torch.nn.Linear(self.info[name]['act_dim'], hidden_size))
            setattr(self, f"predict_state", torch.nn.Linear(hidden_size, self.info[name]['state_dim']))
            setattr(self, f"predict_action", nn.Sequential(
                *([nn.Linear(hidden_size, self.info[name]['act_dim'])] + ([nn.Tanh()] if action_tanh else []))
            ))

            # prompt
            setattr(self, f"prompt_embed_state", torch.nn.Linear(self.info[name]['state_dim'], hidden_size))
            setattr(self, f"prompt_embed_action", torch.nn.Linear(self.info[name]['act_dim'], hidden_size))
            setattr(self, f"prompt_predict_action", torch.nn.Linear(self.info[name]['act_dim'], hidden_size))


        self.prompt_embed_timestep = nn.Embedding(max_ep_len, hidden_size)
        self.prompt_embed_return = torch.nn.Linear(1, hidden_size)


    def forward(self, env_name, states, actions, rewards, returns_to_go, timesteps, attention_mask=None, prompt=None):
        batch_size, seq_length = states.shape[0], states.shape[1]
        if attention_mask is None:
            # attention mask for GPT: 1 if can be attended to, 0 if not
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long)

        if self.special_embedding:
            embed_state = getattr(self, f"{env_name}_embed_state")
            embed_action = getattr(self, f"{env_name}_embed_action")
            predict_state = getattr(self, f"{env_name}_predict_state")
            predict_action = getattr(self, f"{env_name}_predict_action")
        else:
            embed_state = getattr(self, f"embed_state")
            embed_action = getattr(self, f"embed_action")
            predict_state = getattr(self, f"predict_state")
            predict_action = getattr(self, f"predict_action")

        # embed each modality with a different head
        state_embeddings = embed_state(states)
        action_embeddings = embed_action(actions)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # time embeddings are treated similar to positional embeddings
        state_embeddings = state_embeddings + time_embeddings
        action_embeddings = action_embeddings + time_embeddings
        returns_embeddings = returns_embeddings + time_embeddings

        # this makes the sequence look like (R_1, s_1, a_1, R_2, s_2, a_2, ...)
        # which works nice in an autoregressive sense since states predict actions
        stacked_inputs = torch.stack(
            (returns_embeddings, state_embeddings, action_embeddings), dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, 3*seq_length, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # to make the attention mask fit the stacked inputs, have to stack it as well
        stacked_attention_mask = torch.stack(
            (attention_mask, attention_mask, attention_mask), dim=1
        ).permute(0, 2, 1).reshape(batch_size, 3*seq_length)

        # process prompt the same as d-t
        if prompt is not None:
            prompt_states, prompt_actions, prompt_rewards, prompt_dones, prompt_returns_to_go, prompt_timesteps, prompt_attention_mask = prompt
            prompt_seq_length = prompt_states.shape[1]

            if self.special_embedding:
                prompt_embed_state = getattr(self, f"{env_name}_prompt_embed_state")
                prompt_embed_action = getattr(self, f"{env_name}_prompt_embed_action")
            else:
                prompt_embed_state = getattr(self, f"prompt_embed_state")
                prompt_embed_action = getattr(self, f"prompt_embed_action")
            
            prompt_state_embeddings = prompt_embed_state(prompt_states)
            prompt_action_embeddings = prompt_embed_action(prompt_actions)
            if prompt_returns_to_go.shape[1] > prompt_states.shape[1]:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go[:,:-1])
            else:
                prompt_returns_embeddings = self.prompt_embed_return(prompt_returns_to_go)
            prompt_timesteps[prompt_timesteps >= self.max_ep_len] = self.max_ep_len - 1
            prompt_timesteps[prompt_timesteps < 0] = 0
            prompt_time_embeddings = self.prompt_embed_timestep(prompt_timesteps.long())

            prompt_state_embeddings = prompt_state_embeddings + prompt_time_embeddings
            prompt_action_embeddings = prompt_action_embeddings + prompt_time_embeddings
            prompt_returns_embeddings = prompt_returns_embeddings + prompt_time_embeddings

            prompt_stacked_inputs = torch.stack(
                (prompt_returns_embeddings, prompt_state_embeddings, prompt_action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(prompt_states.shape[0], 3 * prompt_seq_length, self.hidden_size)

            # to make the attention mask fit the stacked inputs, have to stack it as well
            prompt_stacked_attention_mask = torch.stack(
                (prompt_attention_mask, prompt_attention_mask, prompt_attention_mask), dim=1
            ).permute(0, 2, 1).reshape(prompt_states.shape[0], 3 * prompt_seq_length)

            # stacked_inputs add prompted sequence
            if prompt_stacked_inputs.shape[0] < batch_size: # if only smaple one prompt
                prompt_stacked_inputs = prompt_stacked_inputs.reshape(1, -1, self.hidden_size)
                prompt_stacked_attention_mask = prompt_stacked_attention_mask.reshape(1, -1)
                stacked_inputs = torch.cat((prompt_stacked_inputs.repeat(batch_size, 1, 1), stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask.repeat(batch_size, 1), stacked_attention_mask), dim=1)
            elif prompt_stacked_inputs.shape[0] > batch_size: # if only smaple more prompt than batch
                prompt_stacked_inputs = prompt_stacked_inputs[0:1]
                prompt_stacked_attention_mask = prompt_stacked_attention_mask[0:1]
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)
            else: # if sample one prompt for each traj in batch
                stacked_inputs = torch.cat((prompt_stacked_inputs, stacked_inputs), dim=1)
                stacked_attention_mask = torch.cat((prompt_stacked_attention_mask, stacked_attention_mask), dim=1)

         # we feed in the input embeddings (not word indices as in NLP) to the model
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        if prompt is None:
            # reshape x so that the second dimension corresponds to the original
            # returns (0), states (1), or actions (2); i.e. x[:,1,t] is the token for s_t
            x = x.reshape(batch_size, seq_length, 3, self.hidden_size).permute(0, 2, 1, 3)
        else:
            x = x.reshape(batch_size, -1, 3, self.hidden_size).permute(0, 2, 1, 3)

        # note here all the prompt are pre-append to x, but when return only return the last [:, -seq_length:, :] corresponding to batch data
        # get predictions
        return_preds = self.predict_return(x[:,2])[:, -seq_length:, :]  # predict next return given state and action
        state_preds = predict_state(x[:,2])[:, -seq_length:, :]    # predict next state given state and action
        action_preds = predict_action(x[:,1])[:, -seq_length:, :]  # predict next action given state

        return state_preds, action_preds, return_preds

    def get_action(self, env_name, states, actions, rewards, returns_to_go, timesteps, info, **kwargs):
        # we don't care about the past rewards in this model
        if info is not None:
            state_dim = info[env_name]['state_dim']
            act_dim = info[env_name]['act_dim']
        else:
            if env_name in info.keys():
                state_dim = info[env_name]['state_dim']
                act_dim = info[env_name]['act_dim']
            else:
                state_dim = info['state_dim']
                act_dim = info['act_dim']

        states = states.reshape(1, -1, state_dim)
        actions = actions.reshape(1, -1, act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if self.max_length is not None:
            states = states[:,-self.max_length:]
            actions = actions[:,-self.max_length:]
            returns_to_go = returns_to_go[:,-self.max_length:]
            timesteps = timesteps[:,-self.max_length:]

            # pad all tokens to sequence length
            attention_mask = torch.cat([torch.zeros(self.max_length-states.shape[1]), torch.ones(states.shape[1])])
            attention_mask = attention_mask.to(dtype=torch.long, device=states.device).reshape(1, -1)
            states = torch.cat(
                [torch.zeros((states.shape[0], self.max_length-states.shape[1], state_dim), device=states.device), states],
                dim=1).to(dtype=torch.float32)
            actions = torch.cat(
                [torch.zeros((actions.shape[0], self.max_length - actions.shape[1], act_dim),
                             device=actions.device), actions],
                dim=1).to(dtype=torch.float32)
            returns_to_go = torch.cat(
                [torch.zeros((returns_to_go.shape[0], self.max_length-returns_to_go.shape[1], 1), device=returns_to_go.device), returns_to_go],
                dim=1).to(dtype=torch.float32)
            timesteps = torch.cat(
                [torch.zeros((timesteps.shape[0], self.max_length-timesteps.shape[1]), device=timesteps.device), timesteps],
                dim=1
            ).to(dtype=torch.long)
        else:
            attention_mask = None

        # Note: prompt within kwargs
        _, action_preds, return_preds = self.forward(
            env_name, states, actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs)

        return action_preds[0,-1]
