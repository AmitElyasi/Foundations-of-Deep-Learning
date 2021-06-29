import torch
import pickle
import random
import numpy as np
from PIL import Image
import matplotlib

matplotlib.use('pdf')
import matplotlib.pyplot as plt

from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url
from torch.utils.model_zoo import load_url as load_state_dict_from_url

TRAIN_BATCH_FILES = [f'./cifar-10-batches-py/data_batch_{i}' for i in range(1, 3)]
TEST_BATCH_FILES = ['./cifar-10-batches-py/test_batch']
LR = 1e-3
WD = 0
EPOCHS = 500
BATCH_SIZE = 16
CUDA_NUM = 0

def plot_loss_vs_epoch(loss_train, question, loss_test=None):
    plt.figure(0)
    plt.plot(loss_train, label='train loss')
    if loss_test:
        plt.plot(loss_test, label='test loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.savefig(f"./plots/{question}_loss_vs_epoch.png")
    plt.close()


def plot_acc_vs_epoch(acc_train, question, acc_test=None):
    plt.figure()
    plt.plot(acc_train, label='train accuracy')
    if acc_test:
        plt.plot(acc_test, label='test accuracy')

    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1])
    plt.savefig(f"./plots/{question}_acc_vs_epoch.png")
    plt.close()


def get_training_device(cuda_num=1):
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{cuda_num}")  # you can continue going on here, like cuda:1 cuda:2....etc.
        print("Running on GPU")
    else:
        device = torch.device("cpu")
        print("Running on CPU")
    return device


def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d


def get_image_from_vector(vec):
    rgb = [vec[:1024], vec[1024:2048], vec[2048:]]
    img_arr = []
    for color in rgb:
        color_arr = []
        for start_ind in range(0, 1024, 32):
            row = color[start_ind:start_ind + 32]
            color_arr.append(row)
        img_arr.append(color_arr)
    img_arr = np.array(img_arr)
    img_arr = np.transpose(img_arr)
    return Image.fromarray(img_arr)


preprocess = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_batch(batch_file):
    batch = unpickle(batch_file)
    data = batch[b"data"]
    labels = batch[b"labels"]

    images = []
    for image_data, label in zip(data, labels):
        image = get_image_from_vector(image_data)
        image = preprocess(image)
        image = np.array(image)
        images.append(image)

    images = np.array(images)
    labels = np.array(labels)
    return images, labels


class Dataset(torch.utils.data.Dataset):
    def __init__(self, load_train=True, random_labels=None):
        self.images, self.labels = self.load_images(load_train, random_labels)

    def load_images(self, load_train=True, random_labels=False):
        files = TRAIN_BATCH_FILES if load_train else TEST_BATCH_FILES
        images = []
        labels = []
        for batch_file in files:
            batch_images, batch_labels = load_batch(batch_file)
            images.extend(batch_images)
            labels.extend(batch_labels)

        if random_labels == "half":
            to_change = len(labels) // 2
            labels = labels[:to_change] + [random.randrange(10) for _ in range(to_change)]
        elif  random_labels == "full":
            labels = [random.randrange(10) for _ in range(len(labels))]
        elif random_labels == "plusone":
            labels = [(l+1)%9 for l in labels]

        images = np.array(images)
        labels = np.array(labels)
        return torch.FloatTensor(images), torch.LongTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


def train(question):
    device = get_training_device(cuda_num=CUDA_NUM)

    model = torch.hub.load('pytorch/vision:v0.9.0', 'inception_v3', pretrained=False)
    # if question == "q1_1":
    #     print("loaded existing model")
    #     model.load_state_dict(torch.load("./models/q1_1.pt"))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

    random_labels = None
    if question == "q1_2":
        random_labels = "full"
    elif question == "q1_3":
        random_labels = "half"
    elif question == "q1_4":
        random_labels = "plusone"

    train_set = Dataset(random_labels=random_labels)
    test_set = Dataset(load_train=False)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

    # Train the model
    acc_test = []
    acc_train = []
    loss_train = []
    loss_test = []

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}, number of batches: {len(train_set)/BATCH_SIZE}")
        cur_loss_train = 0
        cur_loss_test = 0
        cur_test_acc = 0
        cur_train_acc = 0

        total_train_samples = 0
        total_test_samples = 0

        # train
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            # Run the forward pass
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits

            loss = criterion(outputs, labels)
            cur_loss_train += loss.item()

            # Backprop and perform SGD optimisation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            _, prediction = torch.max(outputs.data, 1)
            total = labels.size(0)
            correct = (prediction == labels).sum().item()
            cur_train_acc = correct / total

            total_train_samples += labels.size(0)
        cur_loss_train /= (len(images) + 1)

        # eval on test
        with torch.no_grad():
            model.eval()
            for j, (images, labels) in enumerate(test_loader):
                images = images.float()
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, prediction = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)
                cur_loss_test += loss.item()

                total = labels.size(0)
                correct = (prediction == labels).sum().item()
                cur_test_acc = correct / total

                total_test_samples += labels.size(0)

            cur_loss_test /= (len(images) + 1)

        loss_train.append(cur_loss_train)
        acc_train.append(cur_train_acc)
        loss_test.append(cur_loss_test)
        acc_test.append(cur_test_acc)

        print("Epoch {}, Train Accuracy: {}%, test accuracy: {}% , TrainLoss: {} , Testloss: {}".format(
            epoch,
            acc_train[epoch] * 100,
            acc_test[epoch] * 100,
            loss_train[epoch],
            loss_test[epoch]
        ))

        torch.save(model.state_dict(), f"./models/{question}.pt")
        plot_loss_vs_epoch(loss_train, question, loss_test=loss_test)
        plot_acc_vs_epoch(acc_train, question, acc_test=acc_test)
        if question == "q1_1":
            if acc_train[epoch] == 1 and acc_test[epoch] > 0.9:
                break
        if question == "q1_2" or question == "q1_3":
            if acc_train[epoch] == 1:
                break



# train("q1_1")
# train("q1_2")
# print("DONE Q1 2")
# print("STARTING Q1 3")
# train("q1_3")
# print("DONE Q1 3")
# print("STARTING Q1 4")
# train("q1_4")
# print("DONE Q1 4")
