import math
import statistics
import torch
import model_2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Agent(object):
    def __init__(self, args, eva_dataset, global_model):
        if args.dataset == 'mnist':
            self.model = model_2.MNIST_CNN_Net()
        else:
            self.model = model_2.CIFAR_CNN_Net()
        self.eva_loader = torch.utils.data.DataLoader(eva_dataset, batch_size=args.bs, shuffle=True)
        # self.model.load_state_dict(torch.load(model_path))
        self.model.load_state_dict(global_model)
        self.model.to(device)

    def evaluate(self):
        self.model.to(device)
        self.model.eval()
        correct = 0
        data_size = 0
        for batch_id, batch in enumerate(self.eva_loader):
            data, target = batch
            data_size += data.size()[0]
            data = data.to(device)
            target = target.to(device)

            output = self.model(data)

            pred = output.data.max(1)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

        eval_acc = 100.0 * (float(correct) / float(data_size))

        return eval_acc


def cal_diff(trainer_idx_verifier_loss, loss_dict, trainer_idx):
    beta = 1
    gama = 1
    diff_i = 0

    loss_im = median = statistics.median(trainer_idx_verifier_loss)
    for id, loss_ij in enumerate(trainer_idx_verifier_loss):
        diff_i += gama * abs(loss_ij - loss_dict[trainer_idx]) / math.exp(beta * (loss_ij - loss_im))

    # verifier_loss = np.array(trainer_idx_verifier_loss)
    # verifier_trainer_idx = np.abs(verifier_loss - loss_dict[trainer_idx])
    # verifier_loss = np.abs(verifier_loss - loss_im)
    #
    # diff_trainer_idx = np.sum(verifier_trainer_idx/np.exp(verifier_loss))  # 0.24065

    return diff_i


def cal_agg_weight(diff_trainer_dict, loss_dict, params_dict, trainer_dict, data_num_dict):

    loss_list = list(loss_dict.values())
    sort_loss_list = sorted(loss_list)
    median_loss = sort_loss_list[int(len(loss_list) / 2)]
    abs_loss = [abs(i - median_loss) for i in loss_list]
    sort_abs_loss = sorted(abs_loss)
    median_abs_loss = sort_abs_loss[int(len(abs_loss) / 2)]
    loss_MAD = 3.5 * median_abs_loss

    diff_list = list(diff_trainer_dict.values())
    sort_diff_list = sorted(diff_list)
    median_diff = sort_diff_list[int(len(diff_list)/2)]
    abs_diff = [abs(j - median_diff) for j in diff_list]
    sort_abs_diff = sorted(abs_diff)
    median_abs_diff = sort_abs_diff[int(len(abs_diff)/2)]
    diff_MAD = 3.9 * median_abs_diff

    Di_dict = {}
    test_di = {}
    for trainer_idx in diff_trainer_dict.keys():
        test_di[trainer_idx] = abs(diff_trainer_dict[trainer_idx] - median_diff)
        if abs(diff_trainer_dict[trainer_idx] - median_diff) <= diff_MAD:
            Di_dict[trainer_idx] = diff_MAD - (abs(diff_trainer_dict[trainer_idx] - median_diff))
        else:
            Di_dict[trainer_idx] = 0

    Li_dict = {}
    test_li = {}
    for trainer_id in loss_dict.keys():
        test_li[trainer_id] = abs(loss_dict[trainer_id] - median_loss)
        if abs(loss_dict[trainer_id] - median_loss) <= loss_MAD:
            Li_dict[trainer_id] = loss_MAD - (abs(loss_dict[trainer_id] - median_loss))
        else:
            Li_dict[trainer_id] = 0

    Si_dict = {}
    for id in trainer_dict.keys():
        Si_dict[id] = Di_dict[id] * Li_dict[id]


    weighted_data_num_sum = 0
    for trainer_idx in trainer_dict.keys():
        weighted_data_num_sum += data_num_dict[trainer_idx] * Si_dict[trainer_idx]

    agg_weights = []
    for trainer_idx in trainer_dict.keys():
        agg_weights.append(data_num_dict[trainer_idx] * Si_dict[trainer_idx] / weighted_data_num_sum)

    params = list(params_dict.values())

    return params, agg_weights, test_di, test_li, Si_dict