import cv2
import numpy as np
import json
import glob
import os
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from torchvision import transforms
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time
from skimage import transform, util
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def s_t_frame(video_name, start_frame, end_frame, optical_flow_size=10):  # calculate optical flow
    cap = cv2.VideoCapture(video_name)

    motion_size = end_frame - start_frame
    step_size = int(motion_size / optical_flow_size)

    spatial_frame_index = start_frame + int(motion_size / 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, spatial_frame_index)
    _, spt_frame = cap.read()
    spt_frame = cv2.cvtColor(spt_frame, cv2.COLOR_BGR2RGB)

    flow_stack = np.empty((spt_frame.shape[0], spt_frame.shape[1], optical_flow_size * 2), dtype=np.uint8)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    _, frame1 = cap.read()
    prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    idx = 0
    for i in range(optical_flow_size):
        next_frame_index = start_frame + ((i + 1) * step_size)
        cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_index)
        ret, frame2 = cap.read()
        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        horz = cv2.normalize(flow[..., 0], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        vert = cv2.normalize(flow[..., 1], None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        flow_stack[..., idx] = horz
        flow_stack[..., idx + 1] = vert
        idx += 2
        prev = next.copy()

    return spt_frame, flow_stack


def createH5file(directory, file_name, json_file):  # create HDF5 file
    videos = glob.glob(directory)
    with open(json_file) as file:
        signLanguage_json = json.load(file)

    with h5py.File(file_name, 'w') as hf:
        for i in range(len(videos)):
            video_name = os.path.basename(videos[i])
            print(video_name)
            m = 0
            for k in signLanguage_json:
                if 0 <= k['label'] < 10:
                    if k['url'].split('/')[-1] + '.mp4' == video_name:
                        start_frame = k['start']
                        end_frame = k['end']
                        label = np.array([k['label']], dtype=np.uint8)
                        m += 1
                        print(start_frame, end_frame, m)
                        spt, o_flow = s_t_frame(videos[i], start_frame, end_frame)
                        print(spt.shape, o_flow.shape)
                        spt = transform.resize(spt, (256, 256), anti_aliasing=True)
                        spt = util.img_as_ubyte(spt)
                        resized_flow = np.empty((256, 256, o_flow.shape[2], o_flow.shape[3]), dtype=np.uint8)
                        for i in range(o_flow.shape[3]):
                            temp = np.copy(o_flow[..., i])
                            resized = transform.resize(temp, (256, 256), anti_aliasing=True)
                            resized = util.img_as_ubyte(resized)
                            resized_flow[..., i] = resized
                        handSet_spt = hf.create_dataset(
                            name=video_name + str(m) + '_spt',
                            data=spt,
                            compression="gzip",
                            shape=(spt.shape[0], spt.shape[1], spt.shape[2]),
                            compression_opts=9
                        )
                        handSet_flow = hf.create_dataset(
                            name=video_name + str(m) + '_flow',
                            data=o_flow,
                            compression="gzip",
                            shape=(o_flow.shape[0], o_flow.shape[1], o_flow.shape[2]),
                            compression_opts=9
                        )
                        handSet_label = hf.create_dataset(
                            name=video_name + str(m) + '_label',
                            data=label,
                            compression="gzip",
                            shape=(1,),
                            compression_opts=9
                        )

    return handSet_spt, handSet_flow, handSet_label


class signLanguageDataset(Dataset):  # Dataset function

    def __init__(self, h5_root_dir, train):
        self.h5_root_dir = h5_root_dir
        self.h_dataset = None
        self.train = train
        with h5py.File(self.h5_root_dir, 'r') as file:
            self.total = [str(list(file)[i][: len(list(file)[i]) - 5]) for i in range(len(file)) if
                          (list(file)[i][len(list(file)[i]) - 4:] == 'flow')]

    def __len__(self):
        return len(self.total)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.h_dataset is None:
            self.h_dataset = h5py.File(self.h5_root_dir, 'r')
        flow = self.h_dataset[self.total[idx] + '_flow'][()]
        spt = self.h_dataset[self.total[idx] + '_spt'][()]
        label = self.h_dataset[self.total[idx] + '_label'][()][0]

        spt, flow, label = self.transform(spt, flow, label, self.train)

        sample = {'flow': flow, 'spt': spt, 'label': label}

        return sample

    def transform(self, spt, flow, label, train):
        flow_parse = [flow[..., i] for i in range(flow.shape[2])]

        toPil = transforms.ToPILImage()
        jitter = transforms.ColorJitter()
        spt = toPil(spt)
        flow_parse = [toPil(img) for img in flow_parse]

        if train:
            i, j, h, w = transforms.RandomCrop.get_params(spt, (224, 224))
            spt = TF.crop(spt, i, j, h, w)
            flow_parse = [TF.crop(img, i, j, h, w) for img in flow_parse]

            if np.random.random() > 0.5:
                spt = TF.hflip(spt)
                flow_parse = [TF.hflip(img) for img in flow_parse]

        else:
            spt = TF.center_crop(spt, (224, 224))
            flow_parse = [TF.center_crop(img, (224, 224)) for img in flow_parse]

        spt = jitter(spt)
        spt = TF.to_tensor(spt)
        spt = TF.normalize(spt, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        flow_parse = [TF.to_tensor(img) for img in flow_parse]
        flow_parse = [TF.normalize(img, [0.5], [0.5]) for img in flow_parse]
        flow_parse = [torch.reshape(img, (img.size(1), img.size(2))) for img in flow_parse]
        flow = torch.stack([img for img in flow_parse], dim=0)
        label = torch.tensor(label).long()

        return spt, flow, label


def signLanDataLoader(batch_size):  # Train and Validation Dataloader

    train_dataset = signLanguageDataset('/content/drive/My Drive/signLangTrain.h5', train=True)
    validation_dataset = signLanguageDataset('/content/drive/My Drive/signLangVal.h5', train=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    dataloaders = {'train': train_loader, 'validation': validation_loader}

    return dataloaders


def signLanDataLoaderTest(batch_size):  # Test Dataloader

    test_dataset = signLanguageDataset('/content/drive/My Drive/signLangTest.h5', train=False)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)
    dataloaders = {'test': test_loader}

    return dataloaders


class TwoDCNN(nn.Module):  # (3 x 224 x 224) and (20 x 224 x 224)
    def __init__(self, in_channels, num_classes, drop=0.5):
        super(TwoDCNN, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, kernel_size=7, out_channels=96, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, kernel_size=5, out_channels=256, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=256, kernel_size=3, out_channels=512, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=512, kernel_size=3, out_channels=512, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, kernel_size=3, out_channels=512, stride=1, padding=1)
        self.drop = nn.Dropout(p=drop)
        self.fc6 = nn.Linear(in_features=5 * 5 * 512, out_features=4096)  # 5 x 5
        self.fc7 = nn.Linear(in_features=4096, out_features=2048)
        self.fc8 = nn.Linear(in_features=2048, out_features=num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool(self.relu(self.conv5(x)))
        x = x.view(-1, 5 * 5 * 512)
        x = self.drop(self.relu(self.fc6(x)))
        x = self.drop(self.relu(self.fc7(x)))
        x = self.fc8(x)

        return x


class TwoStreamCNN(nn.Module):  # For early fusion, concatenate last fc layers
    def __init__(self, SpatialCNN, TemporalCNN, num_classes):
        super(TwoStreamCNN, self).__init__()

        self.SpatialCNN = SpatialCNN
        self.TemporalCNN = TemporalCNN

        self.SpatialCNN.fc8 = nn.Identity()
        self.TemporalCNN.fc8 = nn.Identity()

        self.fusionFc = nn.Linear(in_features=2048 + 2048, out_features=num_classes)

    def forward(self, s, t):
        s = self.SpatialCNN(s)
        s = s.view(s.size(0), -1)
        t = self.TemporalCNN(t)
        t = t.view(t.size(0), -1)
        concatenate = torch.cat((s, t), dim=1)
        x = self.fusionFc(concatenate)

        return x


def plotAccLoss(val_acc, val_loss, train_acc, train_loss, information):  # plot function
    fig, ax1 = plt.subplots()

    color1 = 'tab:red'
    color2 = 'tab:blue'
    color3 = 'tab:green'
    color4 = 'tab:orange'

    ax1.set_xlabel('Epoch (Number)')
    ax1.set_ylabel('Accuracy')
    ax1.plot(train_acc, label='Train Accuracy', color=color1)
    ax1.plot(val_acc, label='Validation Accuracy', color=color2)
    ax1.legend(loc='upper right')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss')
    ax2.plot(train_loss, label='Train Loss', color=color3)
    ax2.plot(val_loss, label='Validation Loss', color=color4)

    fig.tight_layout()
    ax2.legend(loc='upper left')
    plt.savefig('/content/drive/My Drive/Ass4 Plots/' + str(information['batch_size']) + str(information['model']) +
                str(information['epoch']) + str(information['drop_spt']) + str(information['drop_temp']) +
                str(information['lr']) + str(information['max']) + '.jpg')
    plt.show()


def train_model(two_stream_cnn, dataloaders, criterion, optimizer, save_path, num_epochs=100, lr_sch=None, plot=False,
                batch_size=None, model=None, drop=None,
                lr=None):  # Train function for spatial, temporal and early fusion

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(two_stream_cnn.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                two_stream_cnn.train()  # Set model to training mode
            else:
                two_stream_cnn.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample in dataloaders[phase]:
                flow, spt, labels = sample['flow'].to(device), sample['spt'].to(device, dtype=torch.float), sample[
                    'label'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    if model == 'Spatial':
                        outputs = two_stream_cnn(spt)
                    elif model == 'Temporal':
                        outputs = two_stream_cnn(flow)
                    else:
                        outputs = two_stream_cnn(spt, flow)

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * spt.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc * 100))

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(two_stream_cnn.state_dict())
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        if lr_sch:
            lr_sch.step()
        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc * 100))

    # load best model weights
    two_stream_cnn.load_state_dict(best_model_wts)
    torch.save(best_model_wts, save_path + str(best_acc) + '.pth')

    if plot:
        information = {
            'batch_size': batch_size,
            'model': model,
            'epoch': num_epochs,
            'drop_spt': drop['spt'],
            'drop_temp': drop['temp'],
            'lr': lr,
            'max': best_acc
        }
        plotAccLoss(val_acc_history, val_loss_history, train_acc_history, train_loss_history, information)

    return two_stream_cnn, val_acc_history, train_acc_history, val_loss_history, train_loss_history


def train_model_late_fusion(model_spatial, model_temporal, dataloaders, criterion, optimizer_spt, optimezer_flow,
                            save_path, num_epochs=100, lr_sch=None, plot=False, batch_size=None, model=None, drop=None,
                            lr=None):  # Train function for late fusion

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts_spatial = copy.deepcopy(model_spatial.state_dict())
    best_model_wts_temporal = copy.deepcopy(model_temporal.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        since = time.time()
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model_spatial.train()  # Set model to training mode
                model_temporal.train()
            else:
                model_spatial.eval()  # Set model to evaluate mode
                model_temporal.eval()

            running_loss = 0.0
            running_corrects = 0
            running_corrects_spatial = 0
            running_corrects_temporal = 0

            # Iterate over data.
            for i, sample in enumerate(dataloaders[phase]):
                flow, spt, labels = sample['flow'].to(device), sample['spt'].to(device, dtype=torch.float), sample[
                    'label'].to(device)

                # zero the parameter gradients
                optimizer_spt.zero_grad()
                optimezer_flow.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs_spatial = model_spatial(spt)
                    outputs_temporal = model_temporal(flow)
                    outputs = outputs_spatial + outputs_temporal

                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    _, preds_spatial = torch.max(outputs_spatial, 1)
                    _, preds_temporal = torch.max(outputs_temporal, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer_spt.step()
                        optimezer_flow.step()

                # statistics
                running_loss += loss.item() * spt.size(0)
                running_corrects += torch.sum(preds == labels.data)
                running_corrects_spatial += torch.sum(preds_spatial == labels.data)
                running_corrects_temporal += torch.sum(preds_temporal == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            epoch_acc_spatial = running_corrects_spatial.double() / len(dataloaders[phase].dataset)
            epoch_acc_temporal = running_corrects_temporal.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f} Acc spatial: {:.4f} Acc flow: {:.4f}'.format(phase, epoch_loss,
                                                                                            epoch_acc * 100,
                                                                                            epoch_acc_spatial * 100,
                                                                                            epoch_acc_temporal * 100))

            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts_spatial = copy.deepcopy(model_spatial.state_dict())
                best_model_wts_temporal = copy.deepcopy(model_temporal.state_dict())
            if phase == 'validation':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)

        if lr_sch:
            lr_sch['spt'].step()
            lr_sch['temporal'].step()
        print()

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc * 100))

    # load best model weights
    model_spatial.load_state_dict(best_model_wts_spatial)
    model_temporal.load_state_dict(best_model_wts_temporal)
    torch.save([best_model_wts_spatial, best_model_wts_temporal], save_path + str(best_acc) + '.pth')

    if plot:
        information = {
            'batch_size': batch_size,
            'model': model,
            'epoch': num_epochs,
            'drop_spt': drop['spt'],
            'drop_temp': drop['temp'],
            'lr': lr,
            'max': best_acc
        }
        plotAccLoss(val_acc_history, val_loss_history, train_acc_history, train_loss_history, information)

    return model_spatial, model_temporal, val_acc_history, train_acc_history, val_loss_history, train_loss_history


def test_model(two_stream_cnn_path, dataloaders, model, save_path,
               num_channels):  # Test function for spatial, temporal and early fusion
    if model == 'Early Fusion':
        model_spatial = TwoDCNN(3, 10)
        model_temporal = TwoDCNN(20, 10)
        two_stream_cnn = TwoStreamCNN(model_spatial, model_temporal, 10)
        model_spatial.to(device)
        model_temporal.to(device)
        two_stream_cnn.to(device)
    else:
        two_stream_cnn = TwoDCNN(num_channels, 10)
        two_stream_cnn.to(device)

    two_stream_cnn.load_state_dict(torch.load(two_stream_cnn_path))

    two_stream_cnn.eval()  # Set model to evaluate mode
    running_corrects = 0
    conf_true = []
    conf_pred = []

    # Iterate over data.
    for sample in dataloaders['test']:
        flow, spt, labels = sample['flow'].to(device), sample['spt'].to(device, dtype=torch.float), sample['label'].to(
            device)
        # zero the parameter gradients

        with torch.no_grad():
            # Get model outputs and calculate loss
            if model == 'Spatial':
                outputs = two_stream_cnn(spt)
            elif model == 'Temporal':
                outputs = two_stream_cnn(flow)
            else:
                outputs = two_stream_cnn(spt, flow)

            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        conf_true[len(conf_true):] = labels.cpu().numpy()
        conf_pred[len(conf_pred):] = preds.cpu().numpy()

    epoch_acc = running_corrects.double() / len(dataloaders['test'].dataset)
    conf_matrix = confusion_matrix(conf_true, conf_pred)
    df_cm = pd.DataFrame(conf_matrix, range(conf_matrix.shape[0]), range(conf_matrix.shape[1]))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True)
    plt.savefig(save_path + model + str(epoch_acc) + '.jpg')
    plt.show()
    print('Acc: {:.4f}'.format(epoch_acc * 100))


def test_model_late_fusion(two_stream_cnn_path, dataloaders, model, save_path):  # Test function for late fusion
    model_spatial = TwoDCNN(3, 10)
    model_temporal = TwoDCNN(20, 10)
    model_spatial.to(device)
    model_temporal.to(device)
    load = torch.load(two_stream_cnn_path)
    model_spatial.load_state_dict(load[0])
    model_temporal.load_state_dict(load[1])

    model_spatial.eval()
    model_temporal.eval()  # Set model to evaluate mode
    running_corrects = 0

    conf_true = []
    conf_pred = []

    # Iterate over data.
    for sample in dataloaders['test']:
        flow, spt, labels = sample['flow'].to(device), sample['spt'].to(device, dtype=torch.float), sample['label'].to(
            device)
        # zero the parameter gradients

        # forward
        # track history if only in train
        with torch.no_grad():
            # Get model outputs and calculate loss
            outputs1 = model_spatial(spt)
            outputs2 = model_temporal(flow)
            outputs = outputs1 + outputs2

            _, preds = torch.max(outputs, 1)

        # statistics
        running_corrects += torch.sum(preds == labels.data)
        conf_true[len(conf_true):] = labels.cpu().numpy()
        conf_pred[len(conf_pred):] = preds.cpu().numpy()

    epoch_acc = running_corrects.double() / len(dataloaders['test'].dataset)
    conf_matrix = confusion_matrix(conf_true, conf_pred)
    df_cm = pd.DataFrame(conf_matrix, range(conf_matrix.shape[0]), range(conf_matrix.shape[1]))
    sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True)
    plt.savefig(save_path + model + str(epoch_acc) + '.jpg')
    plt.show()
    print('Acc: {:.4f}'.format(epoch_acc * 100))


"""**********************************************************************"""  # Experimental functions


def mainEarlyFusion(b, d_s, d_t):
    num_epochs = 100
    batch_size = b
    num_classes = 10
    save_paths = "/content/drive/My Drive/signL"
    drop_spt = d_s
    drop_temp = d_t
    lr = 0.0001
    model = 'Early Fusion'

    spatial_CNN = TwoDCNN(3, num_classes)
    temporal_CNN = TwoDCNN(20, num_classes)
    two_stream_CNN = TwoStreamCNN(spatial_CNN, temporal_CNN, num_classes)
    spatial_CNN.to(device)
    temporal_CNN.to(device)
    two_stream_CNN.to(device)

    dataloaders = signLanDataLoader(batch_size)

    optimizer = optim.Adam(two_stream_CNN.parameters(), lr=lr)
    lr_sch = lr_scheduler.StepLR(optimizer, step_size=40)
    criterion = nn.CrossEntropyLoss().to(device)

    train_model(two_stream_CNN, dataloaders, criterion, optimizer, save_paths,
                num_epochs=num_epochs, lr_sch=lr_sch, plot=True, batch_size=batch_size, model=model,
                drop={'spt': drop_spt, 'temp': drop_temp}, lr=lr)


def mainLateFusion(b, d_s, d_t):
    num_epochs = 100
    batch_size = b
    num_classes = 10
    save_paths = "/content/drive/My Drive/signL"
    drop_spt = d_s
    drop_temp = d_t
    lr = 0.0001
    model = 'Late Fusion'

    spatial_CNN = TwoDCNN(3, num_classes, drop=drop_spt)
    temporal_CNN = TwoDCNN(20, num_classes, drop=drop_temp)
    spatial_CNN.to(device)
    temporal_CNN.to(device)

    dataloaders = signLanDataLoader(batch_size)
    optimizer_spt = optim.Adam(spatial_CNN.parameters(), lr=lr)
    optimizer_temp = optim.Adam(temporal_CNN.parameters(), lr=lr)
    lr_sch = {"spt": lr_scheduler.StepLR(optimizer_spt, step_size=40),
              "temporal": lr_scheduler.StepLR(optimizer_temp, step_size=40)}
    criterion = nn.CrossEntropyLoss().to(device)

    train_model_late_fusion(spatial_CNN, temporal_CNN, dataloaders, criterion, optimizer_spt, optimizer_temp,
                            save_paths,
                            num_epochs=num_epochs, lr_sch=lr_sch, plot=True, batch_size=batch_size, model=model,
                            drop={'spt': drop_spt, 'temp': drop_temp}, lr=lr)


def mainSpatial(b, d_s):
    num_epochs = 100
    batch_size = b
    num_classes = 10
    save_paths = "/content/drive/My Drive/signL"
    drop_spt = d_s
    lr = 0.0001
    model = 'Spatial'

    spatial_CNN = TwoDCNN(3, num_classes, drop_spt)
    spatial_CNN.to(device)

    dataloaders = signLanDataLoader(batch_size)
    optimizer_spt = optim.Adam(spatial_CNN.parameters(), lr=lr)
    lr_sch = lr_scheduler.StepLR(optimizer_spt, step_size=40)
    criterion = nn.CrossEntropyLoss().to(device)

    train_model(spatial_CNN, dataloaders, criterion, optimizer_spt, save_paths,
                num_epochs=num_epochs, lr_sch=lr_sch, plot=True, batch_size=batch_size, model=model,
                drop={'spt': drop_spt, 'temp': 'None'}, lr=lr)


def mainTemporal(b, d_t):
    num_epochs = 100
    batch_size = b
    num_classes = 10
    save_paths = "/content/drive/My Drive/signL"
    drop_temp = d_t
    lr = 0.0001
    model = 'Temporal'

    temporal_CNN = TwoDCNN(20, num_classes, drop=drop_temp)
    temporal_CNN.to(device)

    dataloaders = signLanDataLoader(batch_size)
    optimizer_temp = optim.Adam(temporal_CNN.parameters(), lr=lr)
    lr_sch = lr_scheduler.StepLR(optimizer_temp, step_size=40)
    criterion = nn.CrossEntropyLoss().to(device)

    train_model(temporal_CNN, dataloaders, criterion, optimizer_temp, save_paths,
                num_epochs=num_epochs, lr_sch=lr_sch, plot=True, batch_size=batch_size, model=model,
                drop={'spt': 'None', 'temp': drop_temp}, lr=lr)


def mainTestAndPlotConf(b, l_path, model, s_path, num_channels=None):
    dataloaders = signLanDataLoaderTest(b)
    if model == 'Late Fusion':
        test_model_late_fusion(l_path, dataloaders, model, s_path)
    else:
        test_model(l_path, dataloaders, model, s_path, num_channels)


"""createH5file('/content/drive/My Drive/videos/val/*.mp4', '/content/drive/My Drive/signLanVal.h5', '/content/drive/My Drive/MS-ASL/MSASL_val.json')
createH5file('/content/drive/My Drive/videos/train/*.mp4', '/content/drive/My Drive/signLanTrain.h5', '/content/drive/My Drive/MS-ASL/MSASL_train.json')
createH5file('/content/drive/My Drive/videos/test/*.mp4', '/content/drive/My Drive/signLanTest.h5', '/content/drive/My Drive/MS-ASL/MSASL_test.json')"""

"""mainEarlyFusion(8, 0.2, 0.5)    
mainEarlyFusion(8, 0.4, 0.8)
mainEarlyFusion(16, 0.2, 0.5)    
mainEarlyFusion(16, 0.4, 0.8)
mainLateFusion(8, 0.2, 0.5)
mainLateFusion(8, 0.4, 0.8)
mainLateFusion(16, 0.2, 0.5)
mainLateFusion(16, 0.4, 0.8)
mainSpatial(8, 0.1)
mainSpatial(8, 0.4)
mainSpatial(16, 0.1)
mainSpatial(16, 0.4)
mainTemporal(8, 0.1)
mainTemporal(8, 0.4)
mainTemporal(16, 0.1)
mainTemporal(16, 0.4)"""

"""mainTestAndPlotConf(16, '/content/drive/My Drive/signLtensor(0.3218, device=cuda:0, dtype=torch.float64).pth', 'Early Fusion', '/content/drive/My Drive/Ass4 Plots/confMatrix')"""
