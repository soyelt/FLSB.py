import copy
import random
import torch
import torch.utils.data as data


class Trainer(object):
    def __init__(self, args, model, train_data, eval_data, id=-1):
        self.args = args
        self.trainer_id = id
        self.train_data = train_data
        self.eval_data = eval_data
        self.local_model = model
        # 先将整个训练数据集打乱
        all_range_train = list(range(len(self.train_data)))
        random.shuffle(all_range_train)  # 随机打乱训练数据索引

        # 再按照之前的方式分配
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

        self.train_indices = train_indices
    def set_model_params(self, params):
        self.local_model.load_state_dict(copy.deepcopy(params))

    def local_update(self, device):
        local_model = copy.deepcopy(self.local_model)
        optimizer = torch.optim.SGD(local_model.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        local_model.to(device)
        local_model.train()
        train_loss = 0.0
        correct = 0
        data_size = 0
        Loss = torch.nn.CrossEntropyLoss()

        # start local training
        for batch_id, batch in enumerate(self.train_loader):
            data, target = batch
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = local_model(data)
            loss = Loss(output, target)
            # train_loss += loss.item()
            loss.backward()
            if self.args.dp:
                self.clip_gradients(local_model)
            optimizer.step()
        # ave_train_loss = train_loss / len(self.train_loader)
        # print("Average Training Loss: ", ave_train_loss)
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
