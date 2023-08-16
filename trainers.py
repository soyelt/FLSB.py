import copy
import random
import torch
import torch.utils.data as data


class Trainer(object):
    def __init__(self, args, model, train_data, eval_data, id=-1, malicious_trainers=[]):
        self.args = args
        self.trainer_id = id
        self.train_data = train_data
        self.eval_data = eval_data
        self.local_model = model
        # 先将整个训练数据集打乱
        all_range_train = list(range(len(self.train_data)))
        random.shuffle(all_range_train)  # 随机打乱训练数据索引

        # 分配
        data_len_train = int(len(self.train_data) / self.args.trainer_num)
        train_indices = all_range_train[id * data_len_train: (id + 1) * data_len_train]

        # 同样的方式对评估数据进行打乱和分配
        all_range_eval = list(range(len(self.eval_data)))
        random.shuffle(all_range_eval)  # 随机打乱评估数据索引
        data_len_eval = int(len(self.eval_data) / self.args.trainer_num)
        eval_indices = all_range_eval[id * data_len_eval: (id + 1) * data_len_eval]

        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=args.bs,
                                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                            train_indices))
        self.eval_loader = torch.utils.data.DataLoader(self.eval_data, batch_size=args.bs,
                                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                                           eval_indices))
        # 检查是否应该对此训练者执行后门攻击
        self.backdoor_attack = id in malicious_trainers

        self.train_indices = train_indices

    def _add_backdoor_to_batch(self, data, target, trigger_pattern, target_label):
        """
        Add backdoor trigger to the entire batch of data and change their labels to the target label.
        """
        data[:, :, -5:, -5:] = trigger_pattern  # Add trigger to bottom-right
        target[:] = target_label  # Change label to the target label
        return data, target

    def set_model_params(self, params):
        self.local_model.load_state_dict(copy.deepcopy(params))

    def local_update(self, device, round):
        local_model = copy.deepcopy(self.local_model)
        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        local_model.to(device)
        local_model.train()
        train_loss = 0.0
        correct = 0
        data_size = 0
        Loss = torch.nn.CrossEntropyLoss()

        if round < 1:
            # start local training
            for _ in range(5):
                # Define trigger pattern and target label for backdoor attack
                trigger_pattern = torch.tensor(1.0, device=device)  # An example trigger
                target_label = 7  # or whatever class you want the backdoor to point to

                for batch_id, batch in enumerate(self.train_loader):
                    data, target = batch
                    data, target = data.to(device), target.to(device)

                    if self.backdoor_attack:
                        data, target = self._add_backdoor_to_batch(data, target, trigger_pattern, target_label)

                    optimizer.zero_grad()
                    output = local_model(data)
                    loss = Loss(output, target)
                    loss.backward()
                    if self.args.dp:
                        self.clip_gradients(local_model)
                    optimizer.step()
        else:
            # Define trigger pattern and target label for backdoor attack
            trigger_pattern = torch.tensor(1.0, device=device)  # An example trigger
            target_label = 7  # or whatever class you want the backdoor to point to

            for batch_id, batch in enumerate(self.train_loader):
                data, target = batch
                data, target = data.to(device), target.to(device)

                if self.backdoor_attack:
                    data, target = self._add_backdoor_to_batch(data, target, trigger_pattern, target_label)

                optimizer.zero_grad()
                output = local_model(data)
                loss = Loss(output, target)
                loss.backward()
                if self.args.dp:
                    self.clip_gradients(local_model)
                optimizer.step()

        with torch.no_grad():
            for batch_id, batch in enumerate(self.eval_loader):
                data, target = batch
                data, target = data.to(device), target.to(device)
                data_size += data.size()[0]

                output = local_model(data)

                train_loss += Loss(output, target).item()

                pred = output.data.max(1)[1]
                correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

            acc = 100.0 * (float(correct) / float(data_size))
            train_loss = train_loss / data_size

        return local_model.state_dict(), acc, train_loss, self.train_indices

    def clip_gradients(self, model):
        torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.dp_clip)


# 1. 添加后门触发器和恶意标签的函数
def add_backdoor(x, y, trigger_pattern, target_label):
    """
    Adds a backdoor trigger to the image and modifies the label.
    :param x: Image tensor
    :param y: Original label
    :param trigger_pattern: Trigger pattern tensor (same size as image)
    :param target_label: Malicious target label
    :return: Modified image and label
    """
    x_bd = x.clone()
    x_bd += trigger_pattern
    x_bd = torch.clamp(x_bd, 0, 1)
    y_bd = target_label
    return x_bd, y_bd