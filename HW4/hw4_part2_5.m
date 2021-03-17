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

%% tree
tree = fitctree(train_data',labels_train,'MaxNumSplits',3,'CrossVal','on');
test_labels = predict(tree.Trained{1}, test_data');
%%
tree_acc = sum(labels_test == test_labels)/ 10000;
%% svm
Mdl = fitcecoc(train_data',labels_train);
test_labels = predict(Mdl,test_data');
%% svm accuracy
svm_acc = sum(labels_test == test_labels)/ 10000;