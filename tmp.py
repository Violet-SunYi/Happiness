'''K折交叉验证'''
k = 2
num_val_samples = len(data_train) // k #整数除法
num_epochs = 10
all_scores = []
for i in range(k):
    print('processing fold #', i) # 依次把k分数据中的每一份作为校验数据集
    val_data = data_train[i * num_val_samples : (i+1) * num_val_samples]
    val_targets = data_labels[i* num_val_samples : (i+1) * num_val_samples]

    #把剩下的k-1分数据作为训练数据,如果第i分数据作为校验数据，那么把前i-1份和第i份之后的数据连起来
    partial_train_data = np.concatenate([data_train[: i * num_val_samples],
                                         data_train[(i+1) * num_val_samples:]], axis = 0)
    partial_train_targets = np.concatenate([data_labels[: i * num_val_samples],
                                            data_labels[(i+1) * num_val_samples: ]],
                                          axis = 0)
    #把分割好的训练数据和校验数据输入网络
    network.fit(partial_train_data, partial_train_targets, epochs = num_epochs,
              batch_size = 1, verbose = 0)
    print("evaluate the model")
    test_loss, test_acc = network.evaluate(val_data, val_targets, verbose = 0) # 测试
    all_scores.append(test_acc)
print(all_scores)