import torch.nn as nn
import torch
from utils.Coral import CORAL
from utils import mmd
from model import backbone
from tensorboardX import SummaryWriter
from torch.autograd import Variable

writer = SummaryWriter('./logs')

class Transfer_Net(nn.Module):
    def __init__(self, num_class, base_net='resnet50', transfer_loss='mmd', use_bottleneck=True, bottleneck_width=256, width=1024):
        super(Transfer_Net, self).__init__()
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        self.base_network = backbone.network_dict['resnet50']()
        bottleneck_list = [nn.Linear(self.base_network.output_num(),
                                     bottleneck_width), nn.BatchNorm1d(bottleneck_width), nn.ReLU(), nn.Dropout(0.5)]
        self.bottleneck_layer = nn.Sequential(*bottleneck_list)
        classifier_layer_list = [nn.Linear(bottleneck_width, num_class)]
        self.classifier_layer = nn.Sequential(*classifier_layer_list)

        self.bottleneck_layer[0].weight.data.normal_(0, 0.005)
        self.bottleneck_layer[0].bias.data.fill_(0.1)
        self.classifier_layer[0].weight.data.normal_(0, 0.01)
        self.classifier_layer[0].bias.data.fill_(0.0)
    def forward(self, source, target = None, DA_ON = True):
        if DA_ON == True:
            source = self.base_network(source)
            target = self.base_network(target)
            if self.use_bottleneck:
                source = self.bottleneck_layer(source)
                target = self.bottleneck_layer(target)
            transfer_loss = self.adapt_loss(source, target, self.transfer_loss)

            source_clf = self.classifier_layer(source)
            source_clf = torch.sigmoid(source_clf)
            return source_clf, transfer_loss
        else:
            source = self.base_network(source)
            if self.use_bottleneck:
                source = self.bottleneck_layer(source)
            source_clf = self.classifier_layer(source)
            source_clf = torch.sigmoid(source_clf)
            return source_clf

    def predict(self, x):
        features = self.base_network(x)
        if self.use_bottleneck:
            features = self.bottleneck_layer(features)
        clf = self.classifier_layer(features)
        clf = torch.sigmoid(clf)
        return clf

    def adapt_loss(self, X, Y, adapt_loss):
        """Compute adaptation loss, currently we support mmd and coral

        Arguments:
            X {tensor} -- source matrix
            Y {tensor} -- target matrix
            adapt_loss {string} -- loss type, 'mmd' or 'coral'. You can add your own loss

        Returns:
            [tensor] -- adaptation loss tensor
        """
        if adapt_loss == 'mmd':
            mmd_loss = mmd.MMD_loss()
            loss = mmd_loss(X, Y)
        elif adapt_loss == 'coral':
            loss = CORAL(X, Y)
        else:
            loss = 0
        return loss

if __name__=='__main__':
    """
        需要注释掉transfer_loss
        cmd：tensorboard --logdir ./logs
    """
    model = Transfer_Net(31)
    print(model)
    data_source = Variable(torch.rand(16, 3, 224, 224))
    data_target = Variable(torch.rand(16, 3, 224, 224))
    with writer:
        writer.add_graph(model, input_to_model=(data_source,data_target,))
    print('over')