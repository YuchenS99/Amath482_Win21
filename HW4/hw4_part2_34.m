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

%%
accuracy = [];
index = 1;
for i = 0:8
    label1 = find(labels_train == i);
    digit1 = train_data(:,label1);
    for j = i+1:9
        label2 = find(labels_train == j);
        digit2 = train_data(:,label2);
        [U,S,V,threshold,w,sort1,sort2] = digit_trainer(digit1,...
            digit2, 87);
        label1_test = find(labels_test == i);
        digit1_test = test_data(:,label1_test);
        label2_test = find(labels_test == j);
        digit2_test = test_data(:,label2_test);
        test12 = [digit1_test digit2_test];
        testNum = size(test12, 2);
        testMat = U'* test12;
        pval = w'*testMat;
        ResVec = (pval > threshold);
        trueRes = [i*ones(1,size(digit1_test,2)),...
            j*ones(1,size(digit2_test,2))];
        trueRes = (trueRes == j);
        err = abs(ResVec - trueRes);
        errNum = sum(err);
        sucRate = 1 - errNum/testNum;
        accuracy(index) = sucRate;
        index = index + 1;
    end
end

%%
[max, index_max] = max(accuracy);
[min, index_min] = min(accuracy);

