import os
import numpy as np
import torch
from PIL import Image
import torch.nn.functional
from torch import nn, optim, Tensor
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, TensorDataset, Dataset
from torchsummary import summary
import copy
import torchvision
############################
from hbcnets.trainers import get_data_loader
from setting import args_parser
from hbcnets import datasets, models, utils, trainers, constants
from collections import namedtuple


# with a 10−3 learning rate.
# The weight of each loss function λi is experimentally set as
# λs = 0.55, λc = 0.35, and λu = 0.1
learning_rate=1e-1
rate_s=0.55
rate_c=0.1
rate_u=0.5

#The code is referenced from demo.py
def get_random_image(shape, sample_number):
    random_image = np.random.randint(0, 256, shape).astype(np.uint8)
    random_images = []
    for idx in range(sample_number):
        random_images.append(random_image)
    return random_images
#The code is referenced from the url:https://www.cnblogs.com/zhanjiahui/p/15069514.html
def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)   #C和w*h转置
    gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
    return gram

class RandomImageDataset(Dataset):  # 这是一个Dataset子类
    def __init__(self, transform, sample_number=50000):
        self.X = get_random_image((64, 64, 3), sample_number)
        self.transform = transform
        # self.X = np.vstack(self.X).reshape(-1, 3, 32, 32)
        # self.X = self.X.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        img = Image.fromarray(self.X[index])
        res_X = self.transform(img)
        return res_X

    def __len__(self):
        return len(self.X)

    def get_X(self):
        return self.X

class Classifier(nn.Module):
    def __init__(self, num_classes, with_softmax=False):
        super(Classifier, self).__init__()

        in_dims = (3, constants.IMAGE_SIZE * 2, constants.IMAGE_SIZE * 2, constants.IMAGE_SIZE)
        out_dims = (constants.IMAGE_SIZE * 2, constants.IMAGE_SIZE * 2, constants.IMAGE_SIZE, constants.IMAGE_SIZE)
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.num_classes = num_classes

        for i, (in_dim, out_dim) in enumerate(zip(in_dims, out_dims), 1):
            self.conv_layers.add_module(
                'conv_%d' % i,
                nn.Sequential(
                    nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2),
                    nn.BatchNorm2d(out_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=0.2)
                )
            )

        self.fc_layers.add_module(
            'fc_1',
            nn.Sequential(
                nn.Linear(constants.IMAGE_SIZE * constants.IMAGE_SIZE // 16 * constants.IMAGE_SIZE // 16,
                          constants.IMAGE_SIZE * 2),
                nn.LeakyReLU(),
                nn.Dropout(p=0.5)
            )
        )

        self.fc_layers.add_module(
            'fc_2',
            nn.Sequential(
                nn.Linear(constants.IMAGE_SIZE * 2, num_classes),
            )
        )
        if with_softmax:
            self.fc_layers.add_module('softmax', nn.Sequential(nn.Softmax(dim=1)))

    def forward(self, imgs, feature=False):
        out = imgs
        #获取中间输出conv_layers_output
        conv_layer_output = []
        for i, conv_layer in enumerate(self.conv_layers, start=1):
            out = conv_layer(out)
            conv_layer_output.append(out)
        out = out.flatten(1, -1)
        for fc_layer in self.fc_layers:
            img_features = out
            out = fc_layer(out)
        HBC_outputs = namedtuple("HBCOutputs", ["output1", "output2", "output3", "output4"])
        conv_layers = HBC_outputs(conv_layer_output[0],conv_layer_output[1],conv_layer_output[2],conv_layer_output[3])

        if feature:
            return out, conv_layers, img_features
        return out, conv_layers
#G
class Parameterized(nn.Module):
    def __init__(self, num_inputs, num_classes):
        super(Parameterized, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(num_inputs, num_inputs * 20),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(num_inputs * 20, num_inputs * 10),
            nn.LeakyReLU(),
            nn.Dropout(p=0.25),
            nn.Linear(num_inputs * 10, num_classes)
        )

    def forward(self, inputs):
        return self.model(inputs)

class TransNet(nn.Module):
    def __init__(self, with_softmax=False):
        super(TransNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1),padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),

            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),

            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),

            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1),
            # nn.ReLU(),

            nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(2, 2), padding=1,output_padding=1),
            # nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3, 3), stride=(1, 1), padding=1)
        )
    def forward(self, inputs):
        return self.model(inputs)

def test_model_acc(classifier_F,classifier_G, test_images,test_labels):
    test_data_loader = DataLoader(TensorDataset(torch.Tensor(test_images), torch.Tensor(test_labels).long()),
                                  batch_size=len(test_images) // 50, shuffle=False, drop_last=False)
    eval_acc_1, eval_acc_2, cf_mat_1, cf_mat_2 = utils.evaluate_acc_par(args, classifier_F, classifier_G, test_data_loader,
                                                                        cf_mat=True, roc=False)
    print("\n$$$ Test Accuracy of the BEST model 1 {:.2f}".format(eval_acc_1))
    print("     Confusion Matrix 1:\n", (cf_mat_1 * 100).round(2))
    print("\n$$$ Test Accuracy of the BEST model 2 {:.2f}".format(eval_acc_2))
    print("     Confusion Matrix 2:\n", (cf_mat_2 * 100).round(2))

def train_model_par_P(args, model_T,model_F, model_G, dataset, save_path=None):

    optimizer_F = optim.Adam(model_F.parameters(), lr=args.server_lr)
    optimizer_G = optim.Adam(model_G.parameters(), lr=args.server_lr)
    optimizer_T = optim.sgd(model_T.parameters(),  lr=args.server_lr)

    nll_loss_fn = nn.NLLLoss()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()

    train_epoch_loss_Y, train_epoch_acc_Y, valid_epoch_acc_Y = [], [], []
    train_epoch_loss_S, train_epoch_acc_S, valid_epoch_acc_S = [], [], []
    train_epoch_loss_E = []
    best_valid_acc_Y = 0.

    #style Random


    for epoch in range(args.server_epochs):
        ## Training
        trainloader = trainers.get_data_loader(args, dataset, train=True)
        train_batch_loss_Y, train_batch_acc_Y = [], []
        train_batch_loss_S, train_batch_acc_S = [], []
        train_batch_loss_E = []
        model_F.train()
        for batch_id, (images, labels) in enumerate(trainloader):
            images, labels = images.to(args.device), labels.to(args.device)
            #### Training model_G
            if constants.BETA_S != 0:
                model_F.eval()
                model_G.train()
                optimizer_G.zero_grad()
                out_y = model_F(images)
                out_s = model_G(out_y)
                loss_s = ce_loss_fn(out_s, labels[:, 1])
                loss_s.backward()
                optimizer_G.step()
                model_G.eval()
                model_F.train()
            #### Training model_F
            optimizer_F.zero_grad()
            out_y = model_F(images)
            if constants.SOFTMAX:
                out_y = out_y + 1e-16
                loss_F = nll_loss_fn(torch.log(out_y), labels[:, 0])
            else:
                loss_F = ce_loss_fn(out_y, labels[:, 0])

            if constants.BETA_X != 0:
                if constants.SOFTMAX:
                    sftmx = out_y
                else:
                    sftmx = nn.Softmax(dim=1)(out_y)
                    sftmx = sftmx + 1e-16
                loss_E = -torch.mean(torch.sum(sftmx * torch.log(sftmx), dim=1))
                train_batch_loss_E.append(loss_E.item())

            if constants.BETA_S != 0:
                out_s = model_G(out_y)
                loss_G = ce_loss_fn(out_s, labels[:, 1])
                train_batch_loss_S.append(loss_G.item())
                train_batch_acc_S.append(torch.mean((out_s.max(1)[1] == labels[:, 1]).float()))

            if constants.BETA_X != 0 and constants.BETA_S != 0:
                loss = constants.BETA_Y * loss_F + \
                       constants.BETA_S * loss_G + \
                       constants.BETA_X * loss_E
            elif constants.BETA_S != 0:
                loss = constants.BETA_Y * loss_F + \
                       constants.BETA_S * loss_G
            elif constants.BETA_X != 0:
                loss = constants.BETA_Y * loss_F + \
                       constants.BETA_X * loss_E
            else:
                loss = constants.BETA_Y * loss_F

            loss.backward()
            optimizer_F.step()

            ####
            train_batch_loss_Y.append(loss_F.item())
            train_batch_acc_Y.append(torch.mean((out_y.max(1)[1] == labels[:, 0]).float()))

        train_epoch_loss_Y.append(sum(train_batch_loss_Y) / len(train_batch_loss_Y))
        train_epoch_acc_Y.append(sum(train_batch_acc_Y) / len(train_batch_acc_Y) * 100)
        if train_batch_loss_S:
            train_epoch_loss_S.append(sum(train_batch_loss_S) / len(train_batch_loss_S))
            train_epoch_acc_S.append(sum(train_batch_acc_S) / len(train_batch_acc_S) * 100)
        if train_batch_loss_E:
            train_epoch_loss_E.append(sum(train_batch_loss_E) / len(train_batch_loss_E))

        ## Validation
        validloader = trainers.get_data_loader(args, dataset, train=False)
        acc_Y, acc_S = utils.evaluate_acc_par(args, model_F, model_G, validloader)
        valid_epoch_acc_Y.append(acc_Y)
        valid_epoch_acc_S.append(acc_S)
        #
        # print("_________ Epoch: ", epoch + 1)
        # if save_path:
        #     wy = constants.BETA_Y / (constants.BETA_Y + constants.BETA_S)
        #     ws = constants.BETA_S / (constants.BETA_Y + constants.BETA_S)
        #     current_w_vacc = wy * valid_epoch_acc_Y[-1] + ws * valid_epoch_acc_S[-1]
        #     if current_w_vacc > best_valid_acc_Y:
        #         best_valid_acc_Y = current_w_vacc
        #         torch.save(model_F.state_dict(), save_path + "best_model.pt")
        #         torch.save(model_G.state_dict(), save_path + "best_param_G.pt")
        #         print("**** Best Acc Y on Epoch {} is {:.2f}".format(epoch + 1, best_valid_acc_Y))
        # print("Train Loss Y: {:.5f}, \nTrain Acc Y: {:.2f}".format(train_epoch_loss_Y[-1],
        #                                                            train_epoch_acc_Y[-1]))
        # print("Valid Acc Y: {:.2f}".format(valid_epoch_acc_Y[-1]))
        # if train_epoch_loss_S:
        #     print("Train Loss S: {:.5f}, \nTrain Acc S: {:.2f}".format(train_epoch_loss_S[-1],
        #                                                                train_epoch_acc_S[-1]))
        #     print("Valid Acc S: {:.2f}".format(valid_epoch_acc_S[-1]))
        # if train_epoch_loss_E:
        #     print("Train Loss Entropy: {:.5f}".format(train_epoch_loss_E[-1]))

    return model_F, model_G


def random_transform(image_size=None):
    """ Transforms for style image """
    resize = [transforms.Resize(image_size)] if image_size else []
    transform = transforms.Compose(resize + [transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    return transform




if __name__ == '__main__':
    args = args_parser()  ## Reading the input arguments (see setting.py)
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    ##1. Fetch the datasets
    if constants.DATASET == "utk_face":
        (train_images, train_labels), (test_images, test_labels) = datasets.get_dataset(args)
        train_labels, test_labels = datasets.prepare_labels(train_labels, test_labels)
        train_labels = train_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
        test_labels = test_labels[:, [constants.HONEST, constants.CURIOUS]].astype(int)
        train_dataset = (train_images, train_labels)
    elif constants.DATASET == "celeba":
        (train_images, train_labels), (valid_images, valid_labels), (test_images, test_labels) = datasets.get_dataset(
            args)
        train_labels, valid_labels, test_labels = datasets.prepare_labels(train_labels, test_labels, valid_labels)
        train_dataset = ((train_images, train_labels), (valid_images, valid_labels))
    ## 2. Model
    model_T = TransNet(with_softmax=constants.SOFTMAX)
    model_T.to(args.device)
    summary(model_T, input_size=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE), device=args.device)

    model_F = Classifier(num_classes=constants.K_Y, with_softmax=constants.SOFTMAX)
    model_F.to(args.device)
    model_F.eval()
    summary(model_F, input_size=(3, constants.IMAGE_SIZE, constants.IMAGE_SIZE), device=args.device)

    param_G = Parameterized(num_inputs=constants.K_Y, num_classes=constants.K_S)
    param_G.to(args.device)
    param_G.eval()
    summary(param_G, input_size=(constants.K_Y,), device=args.device)

    ## 3. For logging， laod models and load testing data
    model_F.load_state_dict(torch.load("/home/lichenglong/pycharm_project/HBC_raw/hbcnets/results_reg/utk_face/0_2_3_2_2_5_True/64_0/best_model.pt", map_location=torch.device(args.device)))
    param_G.load_state_dict(torch.load("/home/lichenglong/pycharm_project/HBC_raw/hbcnets/results_reg/utk_face/0_2_3_2_2_5_True/64_0/best_param_G.pt", map_location=torch.device(args.device)))
    # model_F.load_state_dict(torch.load(
    #     "W:\\毕业设计\\honest-but-curious-nets-main\\hbcnets\\results_par\\utk_face\\0_2_3_2_2_5_True\\64_0\\best_model.pt",
    #     map_location=torch.device(args.device)))
    # param_G.load_state_dict(torch.load(
    #      "W:\\毕业设计\\honest-but-curious-nets-main\\hbcnets\\results_par\\utk_face\\0_2_3_2_2_5_True\\64_0\\best_param_G.pt",
    #     map_location=torch.device(args.device)))

    ## 4.train
    model_T.train()
    optim_t = optim.SGD(model_T.parameters(), lr=learning_rate)
    transform = random_transform()
    ce_loss_fn = nn.CrossEntropyLoss().to(args.device)
    mse_loss_fn = nn.MSELoss().to(args.device)
    trainloader = get_data_loader(args, train_dataset, train=True)



    #code is referenced from：https://github.com/eriklindernoren/Fast-Neural-Style-Transfer/blob/master/train.py
    for epoch_index in range(args.server_epochs):
        epoch_metrics = {"content": [], "style": [], "usability":[], "total": []}
        for batch_i, (images, labels) in enumerate(trainloader):
            optim_t.zero_grad()
            images, labels = images.to(args.device), labels.to(args.device)

            random_image_dataset = RandomImageDataset(transform=transform, sample_number=labels.shape[0])
            random_image_dataloader = DataLoader(random_image_dataset, batch_size=200)
            random_images = next(iter(random_image_dataloader)).cuda()
            HBC_output_random, HBC_output_random_layers = model_F(random_images)
            gram_random = [gram_matrix(y) for y in HBC_output_random_layers]

            #content loss
            images_original = images
            images_transformed = model_T(images_original)
            content_loss = mse_loss_fn(images_transformed, images_original)
            #style loss
            HBC_output_transform, HBC_output_transform_layers = model_F(images_transformed)

            gm_y = gram_matrix(HBC_output_transform_layers[3])
            style_loss = mse_loss_fn(gm_y, gram_random[3])
            # for ft_y, gm_s in zip(HBC_output_transform_layers, gram_style):
            #     gm_y = gram_matrix(ft_y)
            #     style_loss += mse_loss_fn(gm_y, gm_s)
            # Usability Loss
            usability_Loss = ce_loss_fn(HBC_output_transform, labels[:, 0])
            #total loss
            total_loss = rate_c*content_loss + rate_s*style_loss + rate_u*usability_Loss
            total_loss.backward()
            optim_t.step()

            epoch_metrics["content"] += [content_loss.item()]
            epoch_metrics["style"] += [style_loss.item()]
            epoch_metrics["usability"] += [usability_Loss.item()]
            epoch_metrics["total"] += [total_loss.item()]

        print("_________ Epoch: ", epoch_index + 1)
        print("Valid Acc content loss: {:.10f}".format(epoch_metrics["content"][-1]))
        print("Valid Acc style loss: {:.10f}".format(epoch_metrics["style"][-1]))
        print("Valid Acc usability loss: {:.10f}".format(epoch_metrics["usability"][-1]))
        print("Valid Acc total loss: {:.10f}".format(epoch_metrics["total"][-1]))

    test_model_acc(model_F, param_G, test_images, test_labels)

    # print("\n$$$ Test Accuracy of the BEST model 1 {:.2f}".format(eval_acc_1))
    # print("     Confusion Matrix 1:\n", (cf_mat_1 * 100).round(2))
    # print("\n$$$ Test Accuracy of the BEST model 2 {:.2f}".format(eval_acc_2))
    # print("     Confusion Matrix 2:\n", (cf_mat_2 * 100).round(2))