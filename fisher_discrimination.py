import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

normal_path = "normal.csv"  # 正确样本路径
failure_path = "failure.csv"  # 错误样本路径
test_path = "test_data.csv"  # 测试样本路径
test_labels_path = 'test_labels.csv'  # 测试样本标签路径


class Fisher_judge(object):
    def __init__(self):
        print('Data loading...')
        self.normal = np.mat(pd.read_csv(normal_path))
        self.failure = np.mat(pd.read_csv(failure_path))
        self.test = np.mat(pd.read_csv(test_path))
        self.test_labels = np.mat(pd.read_csv(test_labels_path))  # 载入数据集
        print('Data loading process complete')
        self.normal_avg = np.reshape(np.mean(self.normal, axis=0), (1, self.normal.shape[1]))
        self.failure_avg = np.reshape(np.mean(self.failure, axis=0), (1, self.failure.shape[1]))
        self.test_avg = np.reshape(np.mean(self.test, axis=0), (1, self.test.shape[1]))  # 计算正确/错误/测试样本集中变量均值

    def fisher(self):
        """
        fisher算法实现
        :param c_1: 
        :param c_2: 
        :return: 
        """
        self.normal = self.normal - self.normal_avg
        self.failure = self.failure - self.failure_avg
        self.nf_avg = self.normal_avg * self.normal.shape[0] + self.failure_avg * self.failure.shape[0]
        self.nf_avg = self.nf_avg / (self.normal.shape[0] + self.failure.shape[0])
        s1 = np.dot(self.normal.T, self.normal)
        s2 = np.dot(self.failure.T, self.failure)
        s = s1 + s2
        w = np.dot(self.nf_avg.T, self.nf_avg)
        temp = np.array(np.dot(s.I, w))
        self.projection, b = np.linalg.eig(temp)

    def judge(self):
        """
        标号0属于normal
        标号1属于failure
        :param sample:
        :param w:
        :param center_1:
        :param center_2:
        :return:
        """
        center_1 = np.dot(self.normal_avg, self.projection.T)
        center_2 = np.dot(self.failure_avg, self.projection.T)
        judge = (center_1 * self.normal.shape[0] + center_2 * self.failure.shape[0]) / (
                    self.normal.shape[0] + self.failure.shape[0])
        pos = np.dot(self.test, self.projection.T)
        pos = np.reshape(pos, (self.test.shape[0], 1))
        predict_labels = []
        if (center_1 < center_2):
            for i in range(len(self.test)):
                if (i % 2000 == 0):
                    print("complete %d / %d" % (i, len(self.test)))
                if (pos[i] < judge):  # 更靠近normal数据
                    predict_labels.append(0)
                    continue
                if (pos[i] >= judge):
                    predict_labels.append(1)
        if (center_1 >= center_2):
            for i in range(len(self.test)):
                if (i % 2000 == 0):
                    print("complete %d / %d" % (i, len(self.test)))
                if (pos[i] > judge):  # 更靠近normal数据
                    predict_labels.append(0)
                    continue
                if (pos[i] <= judge):
                    predict_labels.append(1)
        np.savetxt('predict_labels.csv', predict_labels, fmt='%e', delimiter=',')
        print('Predict labels saved')
        flag = 0
        for i in range(len(self.test_labels)):
            if (self.test_labels[i] == predict_labels[i]):
                flag = flag + 1
        accuracy = flag / len(self.test_labels)
        print('accuracy:%4f' % (accuracy * 100), '%')


if __name__ == '__main__':
    func = Fisher_judge()
    func.fisher()
    func.judge()
