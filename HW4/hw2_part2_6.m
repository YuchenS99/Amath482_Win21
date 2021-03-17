clc; clear all; close all;

% Part 1 - load in and reshape train and test data
[images_train, labels_train] = mnist_parse('train-images-idx3-ubyte',...
                               'train-labels-idx1-ubyte');
[images_test, labels_test] = mnist_parse('t10k-images-idx3-ubyte',...
                            't10k-labels-idx1-ubyte');
% cast the images to double
images_train = im2double(images_train);
images_test = im2double(images_test);
% reshape to one image per column for train data
images_train = reshape(images_train, [28*28, 60000]);
[M, N] = size(images_train);
train_data = images_train - repmat(mean(images_train,2),1,N);
% reshape to one image per column for test data
images_test = reshape(images_test, [28*28, 10000]);
[M, N] = size(images_test);
test_data = images_test - repmat(mean(images_test,2),1,N);

%% easy data
label0 = find(labels_train == 0);
digit0 = train_data(:,label0);
label4 = find(labels_train == 4);
digit4 = train_data(:,label4);
train_data04 = [digit0, digit4];
labels_train04 = [zeros(1,size(digit0,2)),...
    4*ones(1,size(digit4,2))]';
label0_test = find(labels_test == 0);
digit0_test = test_data(:,label0_test);
label4_test = find(labels_test == 4);
digit4_test = test_data(:,label4_test);
test_data04 = [digit0_test, digit4_test];
labels_test04 = [zeros(1,size(digit0_test,2)),...
    4*ones(1,size(digit4_test,2))]';
%% easy tree
tree_04 = fitctree(train_data04',labels_train04,'MaxNumSplits',...
    3,'CrossVal','on');
test_labels = predict(tree_04.Trained{1}, test_data04');
tree_04_acc = sum(labels_test04 == test_labels)/ size(test_labels,1);

%% svm tree
Mdl_04 = fitcecoc(train_data04',labels_train04);
test_labels = predict(Mdl_04,test_data04');
svm_04_acc = sum(labels_test04 == test_labels)/size(test_labels,1);

%% difficult data
label4 = find(labels_train == 4);
digit4 = train_data(:,label4);
label9 = find(labels_train == 9);
digit9 = train_data(:,label9);
train_data49 = [digit4, digit9];
labels_train49 = [4*ones(1,size(digit4,2)),...
    9*ones(1,size(digit9,2))]';
label4_test = find(labels_test == 4);
digit4_test = test_data(:,label4_test);
label9_test = find(labels_test == 9);
digit9_test = test_data(:,label9_test);
test_data49 = [digit4_test, digit9_test];
labels_test49 = [4*ones(1,size(digit4_test,2)),...
    9*ones(1,size(digit9_test,2))]';

%% difficult tree
tree_49 = fitctree(train_data49',labels_train49,'MaxNumSplits',...
    3,'CrossVal','on');
test_labels = predict(tree_49.Trained{1}, test_data49');
tree_49_acc = sum(labels_test49 == test_labels)/ size(test_labels,1);

%% difficult svm
Mdl_49 = fitcecoc(train_data49',labels_train49);
test_labels = predict(Mdl_49,test_data49');
svm_49_acc = sum(labels_test49 == test_labels)/size(test_labels,1);
