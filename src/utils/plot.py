import cv2
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


CLASSES = {0: 'bottle', 1: 'glass', 2: 'packet'}

def transform_img(img, STD=(0.229, 0.224, 0.225), MEAN=(0.485, 0.456, 0.406)):
    img = img.transpose((0, 2, 3, 1))
    img = STD * img + MEAN
    img = np.clip(img, 0, 1)
    return img


def plot(img, targets, preds, n_images, image_size):
    img = transform_img(img).transpose((0, 3, 1, 2)).astype(np.float32)[:n_images] #  First n images from batch
    text_img = (np.ones((n_images, 64, image_size[1], 3))).astype(np.float32)
    for i, t_img in enumerate(text_img):
        
        if targets[i] == preds[i]:
            color = (0, 1, 0)
        else:
            color = (1, 0, 0)

        cv2.putText(t_img, f'TRUE: {CLASSES[int(targets[i])]}', (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA, False)
        cv2.putText(t_img, f'PRED: {CLASSES[int(preds[i])]}', (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA, False)

    # BS H W C ---> BS C H W
    #  0 1 2 3 --->  0 3 1 2
    return np.concatenate((img, text_img.transpose((0, 3, 1, 2))), axis=2)


def plot_confusion_matrix(targets, preds):

    cf_matrix = confusion_matrix(targets, preds)

    fig = plt.figure(figsize=(6.4, 6.4))

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                        cf_matrix.flatten()/np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n" for v1, v2 in
            zip(group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(len(CLASSES), len(CLASSES))

    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_xlabel('\nPredicted Category')
    ax.set_ylabel('Actual Category ')

    ax.xaxis.set_ticklabels(CLASSES.values())
    ax.yaxis.set_ticklabels(CLASSES.values())

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8) / 255.0
    data = (data.reshape(fig.canvas.get_width_height()[::-1] + (3,))[None, ...]).transpose((0, 3, 1, 2))
    return data


def plot_roc_curve(one_hot_targets, preds_probs, classes=CLASSES, save_path='docs/roc.jpg'):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(np.array(one_hot_targets)[:, i], np.array(preds_probs)[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(len(classes)):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve for {classes[i]} (area = {roc_auc[i]:2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")

    plt.savefig(save_path)
    plt.close()


def plot_clf_report(targets, predicts, classes=CLASSES, save_path='docs/clf_report.jpg'):

    # Plot of classification report
    clf_report = classification_report(
                                        targets, 
                                        predicts, 
                                        target_names=classes.values(), 
                                        output_dict=True
                                        )

    ax = sns.heatmap(
                pd.DataFrame(clf_report).iloc[:-1, :].T, 
                annot=True, 
                cmap='Blues', 
                )

    # x axis on top
    ax.xaxis.tick_top() 
    ax.xaxis.set_label_position('top')

    plt.savefig(save_path)
    plt.close()


def plot_errors(er_img, er_targets, er_preds, save_path='docs/prediction_errors.jpg'):
    ax = plot(er_img, er_targets, er_preds, len(er_img), er_img[0].shape[1:])

    grid_img = make_grid(torch.tensor(ax), nrow=7).permute(1, 2, 0)
    plt.imsave(save_path, grid_img.numpy())
    plt.close()