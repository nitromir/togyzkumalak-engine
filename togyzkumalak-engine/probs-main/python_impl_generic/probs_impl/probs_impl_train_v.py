import torch
import torch.utils.data.dataloader
import numpy as np
from collections import defaultdict
import numpy as np

import helpers


def train_value_model(value_model: helpers.BaseValueModel, device, optimizer, experience_replay: helpers.ExperienceReplay, batch_size: int, dataset_drop_ratio: float, num_epochs: int = 3):
    value_model.train()
    # Убеждаемся, что модель на правильном устройстве
    value_model = value_model.to(device)
    helpers.optimizer_to(optimizer, device)

    dataset = []
    for action, env, reward in experience_replay.yield_training_tuples():
        for dataset_row in env.get_rotated_encoded_states_with_symmetry__value_model():
            if dataset_drop_ratio > 1e-5 and np.random.rand() < dataset_drop_ratio:
                continue
            dataset_row.append(reward)
            dataset.append(dataset_row)
    print("Value model dataset length", len(dataset))

    # Проверка на пустой dataset
    if len(dataset) == 0:
        print(f"[WARNING] Empty dataset for value model training! Skipping training.")
        return

    dataloader = helpers.torch_create_dataloader(dataset, device, batch_size=batch_size, shuffle=True, drop_last=True)

    predictions = defaultdict(list)
    
    # Множественные эпохи для лучшего обучения
    for epoch in range(num_epochs):
        epoch_losses = []
        for batch_input in dataloader:
            # Разделяем на inputs и actual_values
            inputs = batch_input[:-1]
            actual_values = batch_input[-1]
            
            # КРИТИЧНО: ВСЕГДА переносим ВСЕ данные на GPU явно
            # Модель использует torch.as_tensor() который создает тензоры на CPU по умолчанию!
            # Поэтому мы ДОЛЖНЫ убедиться, что все уже на GPU
            if isinstance(inputs, tuple):
                # Переносим каждый элемент tuple на GPU
                inputs_gpu = []
                for x in inputs:
                    if torch.is_tensor(x):
                        x_gpu = x.to(device, non_blocking=True)
                    else:
                        # Конвертируем numpy/list в тензор НАПРЯМУЮ на GPU
                        x_gpu = torch.tensor(x, device=device, dtype=torch.float32)
                    inputs_gpu.append(x_gpu)
                inputs = tuple(inputs_gpu)
            elif torch.is_tensor(inputs):
                inputs = (inputs.to(device, non_blocking=True),)
            else:
                # Fallback: конвертируем в tuple тензоров на GPU
                if hasattr(inputs, '__iter__'):
                    inputs = tuple(torch.tensor(x, device=device, dtype=torch.float32) for x in inputs)
                else:
                    inputs = (torch.tensor(inputs, device=device, dtype=torch.float32),)
            
            # actual_values тоже на GPU
            if torch.is_tensor(actual_values):
                actual_values = actual_values.to(device, non_blocking=True)
            else:
                actual_values = torch.tensor(actual_values, device=device, dtype=torch.float32)
            actual_values = actual_values.view((-1, 1)).float()
            
            # Финальная проверка: все inputs должны быть на правильном устройстве
            target_device = device.split(':')[0]  # "cuda" из "cuda:0"
            for i, x in enumerate(inputs):
                if not torch.is_tensor(x):
                    raise RuntimeError(f"Input {i} is not a tensor: {type(x)}")
                if x.device.type != target_device:
                    # Принудительно переносим на правильное устройство
                    inputs = tuple(x.to(device) if j == i else x for j, x in enumerate(inputs))
                    break

            pred_state_value = value_model.forward(*inputs)

            loss = torch.nn.functional.mse_loss(pred_state_value, actual_values)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Gradient clipping для предотвращения gradient explosion
            grad_norm = torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=10.0)
            optimizer.step()

            actual_values = actual_values.detach().cpu().numpy()
            pred_state_value = pred_state_value.detach().cpu().numpy()
            for i in range(len(inputs)):
                actual = float(actual_values[i, 0])
                pred = float(pred_state_value[i, 0])
                predictions[actual].append(pred)

            epoch_losses.append(loss.item())
            helpers.TENSORBOARD.append_scalar('value_loss', loss.item())
            helpers.TENSORBOARD.append_scalar('value_grad_norm', grad_norm.item())
        
        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            print(f"Value model epoch {epoch+1}/{num_epochs}, avg loss: {avg_loss:.4f}")