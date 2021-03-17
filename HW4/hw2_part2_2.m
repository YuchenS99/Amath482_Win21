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
label0 = find(labels_train == 0);
digit0 = train_data(:,label0);
label4 = find(labels_train == 4);
digit4 = train_data(:,label4);
label9 = find(labels_train == 9);
digit9 = train_data(:,label9);
label0_test = find(labels_test == 0);
digit0_test = test_data(:,label0_test);
label4_test = find(labels_test == 4);
digit4_test = test_data(:,label4_test);
label9_test = find(labels_test == 9);
digit9_test = test_data(:,label9_test);
test_data049 = [digit0_test, digit4_test, digit9_test];
labels_test049 = [zeros(1,size(digit0_test,2)),...
    4*ones(1,size(digit4_test,2)),...
    9*ones(1,size(digit9_test,2))]';

%%
[U,S,V,w,vdigit1,vdigit2,vdigit3] = class3_trainer(digit0, digit4,...
    digit9, 87)

sort1 = sort(vdigit1);
sort2 = sort(vdigit2);
sort3 = sort(vdigit3);

threshold1 = get_threshold(sort2,sort3);  
threshold2 = get_threshold(sort3,sort1); 

TestNum = size(test_data049,2); 
TestMat = U'*test_data049;  % PCA projection
pval = w'*TestMat;  % LDA projection

ResVec = zeros(1,TestNum); % Answer by the classifier model
for i = 1:TestNum
    if pval(i) > threshold2
        ResVec(i) = 0; % band1
    elseif pval(i) < threshold1
        ResVec(i) = 4; % band2
    else
        ResVec(i) = 9; % band3
    end
end

err_num = 0;
for i = 1:TestNum
    if (ResVec(i) ~= labels_test049(i))
        err_num = err_num + 1;
    end
end
%%
sucRate = 1-err_num/TestNum