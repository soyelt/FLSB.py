import copy
import random

import torch



class Verifier(object):
    def __init__(self, args, eval_data, model, id=-1):
        self.args = args
        self.verifier_id = id
        self.test_model = model
        self.eval_data = eval_data

        all_range = list(range(len(self.eval_data)))
        random.shuffle(all_range)
        data_len = int(len(self.eval_data) / self.args.verifier_num)
        indices = all_range[id * data_len: (id + 1) * data_len]

        self.eval_loader = torch.utils.data.DataLoader(eval_data, batch_size=self.args.bs,
                                                       sampler=torch.utils.data.sampler.SubsetRandomSampler(indices))

    def test(self, params, device, verifier_idx, trainer_idx, round):
        self.test_model.load_state_dict(copy.deepcopy(params))
        self.test_model.to(device)
        self.test_model.eval()

        Loss = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        data_size = 0
        for batch_id, batch in enumerate(self.eval_loader):
            data, target = batch
            data, target = data.to(device), target.to(device)
            data_size += data.size()[0]

            output = self.test_model(data)
            total_loss += Loss(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        acc = 100.0 * (float(correct) / float(data_size))
        test_loss = total_loss / data_size

        print('----round:{}---- Verifier: {} for trainer {} to Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            round, verifier_idx, trainer_idx, test_loss, correct, data_size, acc))
        return test_loss, acc


