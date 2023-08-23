from models.baseline import TextCNN
import torch
import torch.nn as nn

# 训练模型
def train_and_eval(args, train_loader, valid_loader):
    if args.model_name == 'TextCNN':
        model = TextCNN(args)  # init a model
        if torch.cuda.is_available():  # use cuda
            model.cuda()

    # set criterion and optim
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # train
    for epoch in range(args.num_epochs):
        for i, (texts, labels) in enumerate(train_loader):
            if torch.cuda.is_available():
                texts = texts.long().cuda()
                labels = labels.long().cuda()

            outputs = model(texts)  # model pred
            loss = criterion(outputs, labels)  # cal loss using Adam

            optimizer.zero_grad()  # set grad to zero, otherwise grad will be accumulated in each cycle
            loss.backward()  # backward the loss
            optimizer.step()  # update params
        print(f"Loss after iteration {epoch} is {loss}")

        # eval
        correct = 0
        total = 0
        for texts, labels in valid_loader:
            if torch.cuda.is_available():
                texts = texts.long().cuda()
                labels = labels.long().cuda()

            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Epoch {epoch+1}/{args.num_epochs}, Accuracy on validation set: {correct/total}')
        # TO DO: plot curve of loss


if __name__ == '__main__':
    # test
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.num_filters = 4
    args.model_name = 'TextCNN'
    train_and_eval(args, 1, 1)

